"""Other places the same bytes live.

For every indexed model, ask each extra source for the same repository and compare file by file. A file counts
as available from a source only when the source's published SHA-256 and size equal the address this index
already holds, which came from Hugging Face. The source never supplies the expected hash; it only makes a claim
that clients re-check when they fetch from it.

    python indexer/sources.py [--out .] [--workers 8]

Writes v1/sources/huggingface.co/{owner}/{repo}.json and v1/sources/index.json. No weight bytes are downloaded.
"""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "hologram-index/1 (+https://github.com/humuhumu33/hologram-api)"


def get_json(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 403, 401):
                return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(1.5 * 2 ** attempt)
    return None


def modelscope(repo):
    """ModelScope: same org/name, SHA-256 on every file, CORS open on downloads."""
    data = get_json(f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true")
    files = ((data or {}).get("Data") or {}).get("Files") or []
    out = {}
    for f in files:
        if f.get("Type") == "blob" and f.get("Sha256"):
            out[f["Path"]] = {"sha256": f["Sha256"].lower(), "size": f.get("Size")}
    if not out:
        return None
    quoted = urllib.parse.quote(repo, safe="/")
    return {"kind": "modelscope.cn", "name": "ModelScope", "repo": repo,
            "page": f"https://modelscope.cn/models/{quoted}",
            "resolve": f"https://modelscope.cn/models/{quoted}/resolve/master/", "files": out}


SOURCES = [modelscope]


def pretty(doc):
    return json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def check(root, entry, today):
    repo = entry["name"]
    path = os.path.join(root, "huggingface.co", *repo.split("/"), "latest.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    ours = {f["path"]: f for f in doc["files"]}
    weights = {p for p, f in ours.items() if f.get("weights")}
    result = {
        "name": repo, "manifest": doc["manifest"], "revision": doc["revision"], "checked": today,
        "sources": [{"kind": "huggingface.co", "name": "Hugging Face", "repo": repo,
                     "page": f"https://huggingface.co/{repo}", "files": len(ours), "identical": len(ours),
                     "weights": len(weights), "weights_identical": len(weights)}],
    }
    for source in SOURCES:
        found = source(repo)
        if not found:
            continue
        identical = sorted(p for p, f in ours.items()
                           if p in found["files"]
                           and f["address"] == f"sha256:{found['files'][p]['sha256']}"
                           and (f.get("size") is None or found["files"][p]["size"] == f["size"]))
        if not identical:
            continue
        result["sources"].append({
            "kind": found["kind"], "name": found["name"], "repo": found["repo"], "page": found["page"],
            "resolve": found["resolve"], "files": len(ours), "identical": len(identical),
            "weights": len(weights), "weights_identical": len(weights & set(identical)),
            "missing": sorted(set(ours) - set(identical)),
        })
    # Peer to peer and IPFS: addresses computed from the bytes by indexer/aliases.py for this exact revision.
    alias_path = os.path.join(root, "aliases", "huggingface.co", *repo.split("/")) + ".json"
    if os.path.exists(alias_path):
        alias = json.load(open(alias_path, encoding="utf-8"))
        if alias.get("revision") == doc["revision"]:
            every = {"files": len(ours), "identical": len(ours), "weights": len(weights),
                     "weights_identical": len(weights), "missing": []}
            bt = alias["bittorrent"]
            result["sources"].append({"kind": "bittorrent", "name": "BitTorrent", "p2p": True,
                                      "page": f"https://humuhumu33.github.io/hologram-api/{bt['torrent']}",
                                      "magnet": bt["magnet"], "infohash_v2": bt["infohash_v2"], **every})
            ipfs = alias.get("ipfs") or {}
            if ipfs.get("pinned") and ipfs.get("gateway"):
                result["sources"].append({"kind": "ipfs", "name": "IPFS", "page": ipfs["gateway"],
                                          "gateway": ipfs["gateway"], **every})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = os.path.join(args.out, "v1")
    index = json.load(open(os.path.join(root, "index.json"), encoding="utf-8"))
    today = time.strftime("%Y-%m-%d", time.gmtime())
    started = time.time()
    summary = {}
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(lambda e: check(root, e, today), index["models"]):
            extra = [s for s in result["sources"] if s["kind"] != "huggingface.co"]
            summary[result["name"]] = [{"kind": s["kind"], "identical": s["identical"], "files": s["files"],
                                        "weights_identical": s["weights_identical"], "weights": s["weights"]}
                                       for s in result["sources"]]
            if extra:
                target = os.path.join(root, "sources", "huggingface.co", *result["name"].split("/") ) + ".json"
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(pretty(result))
    with open(os.path.join(root, "sources", "index.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(pretty({"checked": today, "models": dict(sorted(summary.items()))}))
    full = sum(1 for v in summary.values() if any(s["kind"] != "huggingface.co" and s["weights"] and s["weights_identical"] == s["weights"] for s in v))
    partial = sum(1 for v in summary.values() if any(s["kind"] != "huggingface.co" for s in v))
    print(f"{len(summary)} models checked; {partial} also on another source, {full} with every weight file identical; "
          f"{(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
