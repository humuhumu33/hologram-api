#!/usr/bin/env python3
"""Build the Hologram address index from Hugging Face's public metadata.

For every indexed model it writes, under v1/huggingface.co/<owner>/<repo>/:
    <commit>.json   the resolution for that commit — immutable, and a valid lockfile for
                    `hologram-api audit --pin` and `HOLO_VERIFY_PIN`
    latest.json     the same document for the newest indexed commit
and under v1/manifests/:
    <blake3>.json   the canonical manifest bytes, named by their own BLAKE3 address

Addresses: weight files (Git LFS) use the SHA-256 the Hub publishes — no weight bytes are downloaded.
Small files are fetched once, checked against the Hub's Git SHA-1, and addressed by SHA-256.
The canonical manifest is byte-identical to hologram-api's resolver (checked by --selftest).
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import blake3

HUB = "https://huggingface.co"
SELFTEST = ("hf-internal-testing/tiny-random-gpt2",
            "71034c5d8bde858ff824298bdedc65515b97d2b9",
            "blake3:84f5556b153b70dd120468684ff501e3bb4905a251ebff3b32e5bd8989eee76b")
MAX_FILES = 2000
MAX_SMALL_BYTES = 64 << 20
TOKEN = os.environ.get("HF_TOKEN")


def fetch(url, want_json=False, attempts=8):
    headers = {"User-Agent": "hologram-index/1"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    delay = 2.0
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                body = r.read()
                return json.loads(body) if want_json else body
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410):
                raise
            wait = float(e.headers.get("Retry-After") or delay)
            if e.code == 429:
                wait = max(wait, 30.0)
        except (urllib.error.URLError, OSError, TimeoutError):
            wait = delay
        if attempt == attempts - 1:
            raise RuntimeError(f"giving up on {url}")
        time.sleep(min(wait, 300))
        delay = min(delay * 2, 120)


def quote_path(path):
    return "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))


def canonical_bytes(name, revision, files):
    doc = {
        "files": sorted(({"address": f["address"], "path": f["path"], "size": f["size"]} for f in files),
                        key=lambda f: f["path"]),
        "name": name,
        "revision": revision,
        "substrate": "huggingface.co",
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Skip(Exception):
    pass


def resolve(repo, revision="main"):
    info = fetch(f"{HUB}/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true", want_json=True)
    if info.get("private") or info.get("disabled"):
        raise Skip("private or disabled")
    if info.get("gated"):
        raise Skip("gated (not indexed anonymously in v1)")
    siblings = info.get("siblings", [])
    if len(siblings) > MAX_FILES:
        raise Skip(f"{len(siblings)} files exceeds {MAX_FILES}")
    commit = info["sha"]
    files, small = [], []
    for s in siblings:
        entry = {"path": s["rfilename"], "size": s.get("size")}
        lfs = s.get("lfs") or {}
        if lfs.get("sha256"):
            entry["address"] = "sha256:" + lfs["sha256"]
            entry["weights"] = True
        else:
            entry["hub_etag"] = "gitsha1:" + s["blobId"]
            small.append(entry)
        files.append(entry)
    if sum((e["size"] or 0) for e in small) > MAX_SMALL_BYTES:
        raise Skip("small files exceed the ingress cap")

    def hash_small(entry):
        body = fetch(f"{HUB}/{repo}/resolve/{commit}/{quote_path(entry['path'])}")
        git = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        if "gitsha1:" + git != entry["hub_etag"]:
            raise RuntimeError(f"{repo}/{entry['path']}: bytes disagree with the Hub's own Git SHA-1")
        return "sha256:" + hashlib.sha256(body).hexdigest(), len(body)

    ingress = 0
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for entry, (address, n) in zip(small, pool.map(hash_small, small)):
            entry["address"] = address
            ingress += n

    files.sort(key=lambda f: f["path"])
    canonical = canonical_bytes(repo, commit, files)
    manifest = "blake3:" + blake3.blake3(canonical).hexdigest()
    for f in files:
        f["url"] = f"{HUB}/{repo}/resolve/{commit}/{quote_path(f['path'])}"
    resolution = {
        "name": repo,
        "substrate": "huggingface.co",
        "revision": commit,
        "manifest": manifest,
        "files": files,
    }
    return resolution, canonical, ingress


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def pretty(obj):
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def selftest():
    repo, commit, expected = SELFTEST
    resolution, canonical, _ = resolve(repo, commit)
    got = resolution["manifest"]
    assert blake3.blake3(canonical).hexdigest() == got.split(":", 1)[1]
    if got != expected:
        sys.exit(f"SELFTEST FAILED: {got} != {expected} (canonical form drifted from hologram-api)")
    print(f"selftest ok: {repo}@{commit[:12]} -> {got} (identical to hologram-api's Rust resolver)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--limit", type=int, default=200, help="top N models by downloads")
    parser.add_argument("--repos", nargs="*", help="index these repos (in addition to the top N)")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--budget-minutes", type=float, default=300)
    args = parser.parse_args()

    selftest()
    if args.selftest:
        return

    started = time.time()
    root = os.path.join(args.out, "v1")
    listing = fetch(f"{HUB}/api/models?sort=downloads&direction=-1&limit={min(args.limit, 1000)}", want_json=True)
    candidates = [m["id"] for m in listing if not m.get("private")][: args.limit]
    for extra in args.repos or []:
        if extra not in candidates:
            candidates.append(extra)

    index_path = os.path.join(root, "index.json")
    index = {}
    if os.path.exists(index_path):
        index = {e["name"]: e for e in json.load(open(index_path, encoding="utf-8"))["models"]}
    skipped, changed, unchanged, ingress_total = {}, 0, 0, 0

    for position, repo in enumerate(candidates, 1):
        if (time.time() - started) / 60 > args.budget_minutes:
            print(f"time budget reached at {position - 1}/{len(candidates)}; the next run continues")
            break
        try:
            head = fetch(f"{HUB}/api/models/{repo}/revision/main", want_json=True)
            if repo in index and index[repo]["revision"] == head.get("sha"):
                unchanged += 1
                continue
            resolution, canonical, ingress = resolve(repo, "main")
        except Skip as reason:
            skipped[repo] = str(reason)
            continue
        except (urllib.error.HTTPError, RuntimeError, KeyError, ValueError) as error:
            skipped[repo] = f"error: {error}"
            continue

        base = os.path.join(root, "huggingface.co", *repo.split("/"))
        document = pretty(resolution)
        write(os.path.join(base, f"{resolution['revision']}.json"), document)
        write(os.path.join(base, "latest.json"), document)
        write(os.path.join(root, "manifests", resolution["manifest"].split(":", 1)[1] + ".json"), canonical)
        index[repo] = {
            "name": repo,
            "revision": resolution["revision"],
            "manifest": resolution["manifest"],
            "files": len(resolution["files"]),
            "weight_bytes": sum(f["size"] or 0 for f in resolution["files"] if f.get("weights")),
        }
        changed += 1
        ingress_total += ingress
        print(f"[{position}/{len(candidates)}] {repo}@{resolution['revision'][:12]} {resolution['manifest'][:22]}…")

    summary = {
        "substrate": "huggingface.co",
        "models": sorted(index.values(), key=lambda e: e["name"]),
        "skipped": dict(sorted(skipped.items())),
    }
    write(index_path, pretty(summary))
    print(f"indexed {changed} new/changed, {unchanged} unchanged, {len(skipped)} skipped, "
          f"{len(index)} total; small-file ingress {ingress_total / 1e6:.1f} MB; "
          f"{(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
