"""Peer to peer and IPFS addresses for indexed models, computed from the bytes in one streaming pass.

For each file of a model, the pinned Hugging Face URL is streamed once (nothing is written to disk) and teed into:
  - SHA-256 of the whole file, which must equal the index address or the model is refused;
  - BitTorrent v2 (BEP 52): SHA-256 Merkle tree over 16 KiB blocks, pieces root and piece layer;
  - BitTorrent v1 pieces over the same stream with BEP 47 pad files, so the torrent is hybrid;
  - IPFS CIDv1 (optional): `ipfs add --only-hash --cid-version 1 --raw-leaves --chunker size-1048576`.

The torrent's name is the revision and its web seed is Hugging Face's resolve URL, so any client can complete the
download from Hugging Face with zero peers, checking every piece. Aliases are claims: clients still verify SHA-256.

    python indexer/aliases.py --repos hexgrad/Kokoro-82M [--ipfs]

Writes v1/aliases/huggingface.co/{repo}.json and v1/p2p/huggingface.co/{repo}.torrent.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.parse

import requests

BLOCK = 16 * 1024
ZERO = bytes(32)
TRACKERS = ["udp://tracker.opentrackr.org:1337/announce", "udp://open.demonii.com:1337/announce",
            "udp://tracker.torrent.eu.org:451/announce"]
IPFS_PROFILE = "cidv1, raw leaves, 1 MiB fixed chunks, balanced DAG (kubo ipfs add)"


def bencode(value):
    if isinstance(value, int):
        return b"i%de" % value
    if isinstance(value, str):
        value = value.encode()
    if isinstance(value, bytes):
        return b"%d:%s" % (len(value), value)
    if isinstance(value, list):
        return b"l" + b"".join(bencode(v) for v in value) + b"e"
    if isinstance(value, dict):
        items = sorted((k.encode() if isinstance(k, str) else k, v) for k, v in value.items())
        return b"d" + b"".join(bencode(k) + bencode(v) for k, v in items) + b"e"
    raise TypeError(type(value))


def merkle(layer, pad):
    """Root of a layer padded to a power of two with `pad` (the hash of an empty subtree at this height)."""
    n = 1
    while n < len(layer):
        n *= 2
    layer = layer + [pad] * (n - len(layer))
    while len(layer) > 1:
        pad = hashlib.sha256(pad + pad).digest()
        layer = [hashlib.sha256(layer[i] + layer[i + 1]).digest() for i in range(0, len(layer), 2)]
    return layer[0]


def zero_subtree(height):
    h = ZERO
    for _ in range(height):
        h = hashlib.sha256(h + h).digest()
    return h


def piece_length_for(total):
    length = 256 * 1024
    while total / length > 2000 and length < 16 * 1024 * 1024:
        length *= 2
    return length


class V1Pieces:
    def __init__(self, piece_length):
        self.len, self.buf, self.pieces = piece_length, bytearray(), []

    def update(self, data):
        self.buf += data
        while len(self.buf) >= self.len:
            self.pieces.append(hashlib.sha1(bytes(self.buf[: self.len])).digest())
            del self.buf[: self.len]

    def pad(self):
        """Zero fill to the next piece boundary; returns the pad length."""
        n = (-len(self.buf)) % self.len
        if n:
            self.update(bytes(n))
        return n

    def finish(self):
        if self.buf:
            self.pieces.append(hashlib.sha1(bytes(self.buf)).digest())
        return b"".join(self.pieces)


def stream_file(url, size, expected, v1, piece_length, ipfs):
    sha = hashlib.sha256()
    leaves, pending = [], bytearray()
    proc = subprocess.Popen(["ipfs", "add", "--only-hash", "--cid-version", "1", "--raw-leaves", "--chunker",
                             "size-1048576", "-Q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL) if ipfs else None
    received = 0
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "hologram-index/1"}) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            received += len(chunk)
            sha.update(chunk)
            v1.update(chunk)
            if proc:
                proc.stdin.write(chunk)
            pending += chunk
            while len(pending) >= BLOCK:
                leaves.append(hashlib.sha256(bytes(pending[:BLOCK])).digest())
                del pending[:BLOCK]
    if pending:
        leaves.append(hashlib.sha256(bytes(pending)).digest())
    cid = None
    if proc:
        proc.stdin.close()
        cid = proc.stdout.read().decode().strip()
        proc.wait()
    if received != size or f"sha256:{sha.hexdigest()}" != expected:
        raise ValueError(f"{url}: bytes do not match the index address")
    # BEP 52: leaves beyond the file end are zero; piece layer only for files larger than one piece.
    height = (piece_length // BLOCK).bit_length() - 1
    per_piece = piece_length // BLOCK
    layer = None
    if size > piece_length:
        layer = []
        for i in range(0, len(leaves), per_piece):
            part = leaves[i:i + per_piece]
            layer.append(merkle(part + [ZERO] * (per_piece - len(part)), ZERO))
        root = merkle(layer, zero_subtree(height))
    else:
        root = merkle(leaves, ZERO) if leaves else None
    return root, (b"".join(layer) if layer else None), cid


def run(root_dir, repo, ipfs):
    doc = json.load(open(os.path.join(root_dir, "huggingface.co", *repo.split("/"), "latest.json"), encoding="utf-8"))
    files = sorted((f for f in doc["files"] if f.get("size")), key=lambda f: f["path"].encode())
    total = sum(f["size"] for f in files)
    piece_length = piece_length_for(total)
    v1 = V1Pieces(piece_length)
    tree, v1_files, layers, out = {}, [], {}, {}
    started = time.time()
    for i, f in enumerate(files):
        for attempt in range(4):
            state = (bytearray(v1.buf), list(v1.pieces))
            try:
                root, layer, cid = stream_file(f["url"], f["size"], f["address"], v1, piece_length, ipfs)
                break
            except (requests.RequestException, ConnectionError) as error:
                v1.buf, v1.pieces = state  # the stream is shared across files: undo this file's partial bytes
                if attempt == 3:
                    raise
                print(f"  retry {f['path']}: {error.__class__.__name__}")
                time.sleep(2 * 2 ** attempt)
        node = tree
        parts = f["path"].split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {"": {"length": f["size"], **({"pieces root": root} if root else {})}}
        if layer:
            layers[root] = layer
        v1_files.append({"length": f["size"], "path": parts})
        pad = v1.pad()  # hybrid layout: every file, the last included, ends on a piece boundary
        if pad:
            v1_files.append({"length": pad, "path": [".pad", str(pad)], "attr": "p"})
        out[f["path"]] = {"btv2": root.hex() if root else None, **({"ipfs": cid} if cid else {})}
    webseed = f"https://huggingface.co/{repo}/resolve/"
    info = {"name": doc["revision"], "piece length": piece_length, "meta version": 2, "file tree": tree,
            "files": v1_files, "pieces": v1.finish()}
    torrent = {"announce": TRACKERS[0], "announce-list": [[t] for t in TRACKERS], "created by": "hologram-index",
               "comment": f"{repo} at {doc['revision']}, manifest {doc['manifest']}", "info": info,
               "piece layers": layers, "url-list": [webseed]}
    encoded_info = bencode(info)
    ih1, ih2 = hashlib.sha1(encoded_info).hexdigest(), hashlib.sha256(encoded_info).hexdigest()
    torrent_rel = f"v1/p2p/huggingface.co/{repo}.torrent"
    os.makedirs(os.path.dirname(os.path.join(root_dir, "..", torrent_rel)), exist_ok=True)
    with open(os.path.join(root_dir, "..", torrent_rel), "wb") as fh:
        fh.write(bencode(torrent))
    magnet = (f"magnet:?xt=urn:btih:{ih1}&xt=urn:btmh:1220{ih2}&dn={urllib.parse.quote(repo)}"
              f"&ws={urllib.parse.quote(webseed, safe='')}" + "".join(f"&tr={urllib.parse.quote(t, safe='')}" for t in TRACKERS))
    result = {
        "name": repo, "revision": doc["revision"], "manifest": doc["manifest"],
        "computed": time.strftime("%Y-%m-%d", time.gmtime()), "bytes": total,
        "bittorrent": {"infohash_v1": ih1, "infohash_v2": ih2, "piece_length": piece_length,
                       "torrent": torrent_rel, "magnet": magnet, "webseed": webseed},
        **({"ipfs": {"profile": IPFS_PROFILE, "pinned": False}} if ipfs else {}),
        "files": out,
    }
    target = os.path.join(root_dir, "aliases", "huggingface.co", *repo.split("/")) + ".json"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(result, indent=2) + "\n")
    print(f"{repo}: {len(files)} files, {total / 1e9:.2f} GB, piece {piece_length // 1024} KiB, "
          f"v2 {ih2[:16]}…, {time.time() - started:.0f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--repos", nargs="+", required=True)
    parser.add_argument("--ipfs", action="store_true", help="also compute IPFS CIDs (needs the ipfs binary)")
    args = parser.parse_args()
    if args.ipfs and not shutil.which("ipfs"):
        raise SystemExit("--ipfs needs the ipfs (kubo) binary on PATH")
    for repo in args.repos:
        run(os.path.join(args.out, "v1"), repo, args.ipfs)


if __name__ == "__main__":
    main()
