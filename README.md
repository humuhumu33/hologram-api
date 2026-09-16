# Hologram Index

Self-verifying addresses for open model weights. A static, append-only index any agent can read with one
`curl`, served for free by GitHub Pages and jsDelivr.

```bash
curl -s https://humuhumu33.github.io/hologram-index/v1/huggingface.co/sentence-transformers/all-MiniLM-L6-v2/latest.json
```

Each document lists every file in the model with its SHA-256 address and its normal Hugging Face URL.
Download the file however you already do, hash it, compare. The document doubles as a lockfile for
`holo-verify` (`HOLO_VERIFY_PIN`) and `hologram-api audit --pin`.

## Why it is shaped like this

- **Answers never change.** A commit-pinned resolution is a fact, so it is a static file, not a service.
  Static files are served for free at any scale, with no servers to keep up.
- **No host is trusted.** Every file carries its SHA-256 and every manifest is named by its own BLAKE3,
  so a mirror that lies is caught by the client. That is why the same bytes can be served from several
  free CDNs at once.
- **History is public.** This is a Git repository; every indexing run is a commit. Anyone can clone it,
  and a silent change to a published address would show up in the history.
- **No weight bytes are downloaded to build it.** Weight files take the SHA-256 the Hub publishes; small
  files are fetched once, checked against the Hub's own Git SHA-1, and addressed by SHA-256.

## Layout

| Path | What |
|---|---|
| `v1/huggingface.co/<owner>/<repo>/latest.json` | newest indexed commit |
| `v1/huggingface.co/<owner>/<repo>/<commit>.json` | pinned commit — immutable |
| `v1/manifests/<blake3>.json` | canonical manifest bytes, named by their own hash |
| `v1/index.json` | every indexed model, and every repo skipped with the reason |
| `llms.txt`, `capabilities.json` | agent guide and machine-readable description |
| `indexer/index.py` | the indexer; `--selftest` checks the canonical form against hologram-api |

## Coverage

Top Hugging Face models by downloads, public and ungated, refreshed daily by
[`.github/workflows/index.yml`](.github/workflows/index.yml). Gated models need an authenticated indexer
and are not included yet. `latest.json` can trail a repo by up to a day plus CDN caching; `<commit>.json`
never changes.

## What this does and does not prove

It proves you received the bytes that were published at a commit. It does not prove those bytes are
safe — a malicious upload is indexed as faithfully as an honest one. The public history makes a silent
change detectable; signing is not implemented yet. Data is Hugging Face's public metadata; published
addresses are cached permanently by jsDelivr and cannot be withdrawn from it.
