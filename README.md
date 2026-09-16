# Hologram API

**Content-address and verify any bytes, right where they already are.**
An address is the hash of the bytes, so anyone can recompute it and nobody has to be trusted.

→ **https://humuhumu33.github.io/hologram-api/** — drop a file to address or verify it; nothing is uploaded.
→ Agents: [`llms.txt`](https://humuhumu33.github.io/hologram-api/llms.txt) · [`capabilities.json`](https://humuhumu33.github.io/hologram-api/capabilities.json)

```js
import { address, verify, fetchVerified, resolve } from "https://humuhumu33.github.io/hologram-api/hologram.js";

await address(file)                          // { sha256: "sha256:…", blake3: "blake3:…", size }
await verify(file, "sha256:…")               // { ok, expected, received }
await fetchVerified(url, "sha256:…")         // bytes, or throws Refused
await resolve("sentence-transformers/all-MiniLM-L6-v2")   // every file of a model, addressed
```

```bash
sha256sum file.bin                                   # address, anywhere, no install
echo "<hex>  file.bin" | sha256sum -c -              # verify
curl -s https://humuhumu33.github.io/hologram-api/v1/huggingface.co/<owner>/<repo>/latest.json   # resolve a model
```

## Why there is no server

Addressing and verifying are pure functions of the bytes. A server that hashes your data makes you upload it
(slow, and it sees your data) and then asks you to trust its answer — which defeats verification. So the
functions run where the bytes are, in the browser, the shell or the agent. The only published data is the
model index, and it is static: commit-pinned answers never change, so they are files served for free by
GitHub Pages and jsDelivr at any scale, and every document is named by its own hash so no host needs trusting.

## Layout

| Path | What |
|---|---|
| `index.html` | the product page — address, verify, resolve, recipes |
| `hologram.js` | the client API; vendored `hash-wasm` (MIT) for BLAKE3/SHA in the browser |
| `llms.txt`, `capabilities.json` | agent guide and machine-readable description |
| `v1/huggingface.co/<owner>/<repo>/{latest,<commit>}.json` | model documents; each is also a lockfile |
| `v1/manifests/<blake3>.json` | canonical manifests, named by their own hash |
| `v1/index.json` | every indexed model, and every repo skipped with the reason |
| `indexer/index.py` | daily indexer; `--selftest` checks the canonical form against hologram-api-server |
| `tests/hologram.test.mjs` | client tests: vectors, streaming, tamper refusal, live resolve |

## Coverage and limits

The model index covers the most-downloaded public, ungated Hugging Face models (961 at the first daily run)
and refreshes daily; gated models need an authenticated indexer and are not included yet. `latest.json` can
trail a repo by up to a day plus CDN caching; `<commit>.json` never changes. Verification proves you have the
bytes that were addressed, not that those bytes are safe. Published documents are cached permanently by
jsDelivr and cannot be withdrawn from it.

Storing bytes by address (`PUT`/`GET /v1/a`) and shared manifests need a stateful service; that part of
Hologram API runs on hologram-api-server (Rust, built on Hologram Live) and is not yet publicly hosted.

Apache-2.0.
