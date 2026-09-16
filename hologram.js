// Hologram API — content-address and verify any bytes, where the bytes already are.
//
//   import { address, verify, fetchVerified, resolve } from "https://humuhumu33.github.io/hologram-api/hologram.js";
//
//   await address(file)                        → { sha256: "sha256:…", blake3: "blake3:…", size }
//   await verify(file, "sha256:…")             → { ok: true|false, expected, received, size }
//   await fetchVerified(url, "sha256:…")       → Uint8Array, or throws Refused before you touch a byte
//   await resolve("owner/repo")                → every file of a Hugging Face model with its address
//
// Runs in browsers, Node ≥ 18, Deno, Bun and Workers. Nothing is uploaded: hashing happens here, so the
// answer does not depend on trusting any server — including this one. Apache-2.0.

import { createBLAKE3, createSHA1, createSHA256 } from "./vendor/hash-wasm/index.esm.min.js";

export const VERSION = "1.0.0";

export const HOSTS = [
  "https://humuhumu33.github.io/hologram-api",
  "https://cdn.jsdelivr.net/gh/humuhumu33/hologram-api@main",
];

const PATTERNS = {
  sha256: /^[0-9a-f]{64}$/,
  blake3: /^[0-9a-f]{64}$/,
  gitsha1: /^[0-9a-f]{40}$/,
};

/** A stable, machine-readable refusal. `code` never changes meaning. */
export class Refused extends Error {
  constructor(code, message, { expected, received, next } = {}) {
    super(`${code}: ${message}`);
    this.name = "Refused";
    this.code = code;
    this.expected = expected;
    this.received = received;
    this.next = next;
  }
  toJSON() {
    return { error: { code: this.code, message: this.message, expected: this.expected, received: this.received, next: this.next } };
  }
}

/** Parse `sha256:<hex>`, `blake3:<hex>` or `gitsha1:<hex>`. */
export function parseAddress(value) {
  const [algorithm, digest, extra] = String(value ?? "").toLowerCase().split(":");
  if (extra !== undefined || !PATTERNS[algorithm] || !PATTERNS[algorithm].test(digest ?? "")) {
    throw new Refused("BAD_ADDRESS", `not an address: ${JSON.stringify(value)}`, {
      next: "use sha256:<64 hex>, blake3:<64 hex> or gitsha1:<40 hex>",
    });
  }
  return { algorithm, digest };
}

async function* chunks(input) {
  if (input == null) throw new Refused("BAD_INPUT", "nothing to hash");
  if (typeof input === "string") { yield new TextEncoder().encode(input); return; }
  if (input instanceof Uint8Array) { yield input; return; }
  if (ArrayBuffer.isView(input)) { yield new Uint8Array(input.buffer, input.byteOffset, input.byteLength); return; }
  if (input instanceof ArrayBuffer) { yield new Uint8Array(input); return; }
  if (typeof Response !== "undefined" && input instanceof Response) {
    if (!input.body) return;
    input = input.body;
  }
  if (typeof input.stream === "function") input = input.stream(); // Blob, File
  if (typeof input.getReader === "function") {
    const reader = input.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) return;
        yield value;
      }
    } finally {
      reader.releaseLock?.();
    }
  }
  if (typeof input[Symbol.asyncIterator] === "function" || typeof input[Symbol.iterator] === "function") {
    for await (const chunk of input) yield chunk instanceof Uint8Array ? chunk : new TextEncoder().encode(String(chunk));
    return;
  }
  throw new Refused("BAD_INPUT", "input must be bytes, a string, a Blob/File, a Response or a stream");
}

function knownSize(input) {
  if (typeof input === "string") return new TextEncoder().encode(input).byteLength;
  if (ArrayBuffer.isView(input) || input instanceof ArrayBuffer) return input.byteLength;
  if (input && typeof input.size === "number") return input.size;
  return null;
}

async function hashers(algorithms, size) {
  const made = {};
  for (const algorithm of algorithms) {
    if (algorithm === "sha256") made.sha256 = await createSHA256();
    else if (algorithm === "blake3") made.blake3 = await createBLAKE3();
    else if (algorithm === "gitsha1") {
      if (size == null) throw new Refused("UNVERIFIABLE", "gitsha1 needs the content length before hashing");
      made.gitsha1 = await createSHA1();
    } else throw new Refused("BAD_ADDRESS", `unknown algorithm ${algorithm}`);
  }
  for (const [algorithm, h] of Object.entries(made)) {
    h.init();
    if (algorithm === "gitsha1") h.update(new TextEncoder().encode(`blob ${size}\0`));
  }
  return made;
}

/**
 * Content-address bytes. One pass computes every requested algorithm.
 * @returns {Promise<{sha256?: string, blake3?: string, gitsha1?: string, size: number}>}
 */
export async function address(input, algorithms = ["sha256", "blake3"]) {
  const size = knownSize(input);
  const hs = await hashers(algorithms, size);
  let total = 0;
  for await (const chunk of chunks(input)) {
    total += chunk.byteLength;
    for (const h of Object.values(hs)) h.update(chunk);
  }
  const out = { size: total };
  for (const [algorithm, h] of Object.entries(hs)) out[algorithm] = `${algorithm}:${h.digest("hex")}`;
  return out;
}

/** Compare bytes against an address. Never throws on a mismatch — read `ok`. */
export async function verify(input, expected) {
  const { algorithm, digest } = parseAddress(expected);
  const result = await address(input, [algorithm]);
  const received = result[algorithm];
  return { ok: received === `${algorithm}:${digest}`, expected: `${algorithm}:${digest}`, received, size: result.size };
}

/** Like `verify`, but throws `Refused` (code ADDRESS_MISMATCH) when the bytes are not the bytes addressed. */
export async function check(input, expected) {
  const result = await verify(input, expected);
  if (!result.ok) {
    throw new Refused("ADDRESS_MISMATCH", "the bytes received are not the bytes addressed", {
      expected: result.expected,
      received: result.received,
      next: "discard the bytes; fetch from another source with the same address",
    });
  }
  return result;
}

/**
 * Download a URL and return its bytes only if they match the address. The expected address must come from
 * somewhere the download did not: a lockfile, `resolve`, another agent.
 */
export async function fetchVerified(url, expected, init) {
  const { algorithm, digest } = parseAddress(expected);
  const response = await fetch(url, init);
  if (!response.ok) throw new Refused("UPSTREAM_ERROR", `${url} answered ${response.status}`);
  const declared = Number(response.headers.get("content-length"));
  const size = algorithm === "gitsha1" ? (Number.isFinite(declared) && declared > 0 ? declared : null) : null;
  const hs = await hashers([algorithm], size);
  const parts = [];
  let total = 0;
  for await (const chunk of chunks(response)) {
    hs[algorithm].update(chunk);
    parts.push(chunk);
    total += chunk.byteLength;
  }
  const received = `${algorithm}:${hs[algorithm].digest("hex")}`;
  if (received !== `${algorithm}:${digest}`) {
    throw new Refused("ADDRESS_MISMATCH", "the bytes received are not the bytes addressed", {
      expected: `${algorithm}:${digest}`,
      received,
      next: "discard the bytes; fetch from another source with the same address",
    });
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.byteLength; }
  return bytes;
}

const byCodePoint = (a, b) => {
  const x = [...a], y = [...b];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (d) return d;
  }
  return x.length - y.length;
};

/** Canonical manifest bytes — identical to hologram-api-server and the indexer. */
export function canonicalManifest(doc) {
  const files = doc.files
    .map((f) => ({ address: f.address, path: f.path, size: f.size ?? null }))
    .sort((a, b) => byCodePoint(a.path, b.path));
  return new TextEncoder().encode(JSON.stringify({ files, name: doc.name, revision: doc.revision, substrate: doc.substrate }));
}

/**
 * Every file of a Hugging Face model with its address. The document is checked against its own manifest
 * address; pass `{ manifest: "blake3:…" }` to also require a specific, previously pinned manifest.
 */
export async function resolve(repo, { revision = "latest", manifest, hosts = HOSTS } = {}) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(repo)) {
    throw new Refused("NOT_FOUND", `not a model repository name: ${JSON.stringify(repo)}`, { next: "use owner/name" });
  }
  let lastError;
  for (const host of hosts) {
    try {
      const response = await fetch(`${host}/v1/huggingface.co/${repo}/${revision}.json`);
      if (response.status === 404) { lastError = new Refused("NOT_FOUND", `${repo} is not indexed yet`, { next: "the index covers the most-downloaded public models and refreshes daily" }); continue; }
      if (!response.ok) { lastError = new Refused("UPSTREAM_ERROR", `${host} answered ${response.status}`); continue; }
      const doc = await response.json();
      const self = await address(canonicalManifest(doc), ["blake3"]);
      if (self.blake3 !== doc.manifest) {
        lastError = new Refused("ADDRESS_MISMATCH", `${host} served a document that does not match its own manifest address`, { expected: doc.manifest, received: self.blake3 });
        continue;
      }
      if (manifest && parseAddress(manifest) && manifest.toLowerCase() !== doc.manifest) {
        throw new Refused("ADDRESS_MISMATCH", "this model no longer has the pinned manifest", { expected: manifest, received: doc.manifest, next: "the name now points at different bytes; keep your pinned copy" });
      }
      return doc;
    } catch (error) {
      if (error instanceof Refused && error.code === "ADDRESS_MISMATCH" && manifest) throw error;
      lastError = error instanceof Refused ? error : new Refused("UPSTREAM_ERROR", `${host}: ${error.message}`);
    }
  }
  throw lastError;
}

/**
 * Other places the same bytes live. Each source lists which files it serves with the identical SHA-256; the
 * addresses still come from `resolve`, so fetch from any source with `fetchVerified(source.resolve + path, file.address)`.
 */
export async function sources(repo, { hosts = HOSTS } = {}) {
  for (const host of hosts) {
    const response = await fetch(`${host}/v1/sources/huggingface.co/${repo}.json`).catch(() => null);
    if (response?.ok) return (await response.json()).sources;
    if (response?.status === 404) break;
  }
  return [{ kind: "huggingface.co", name: "Hugging Face", repo, page: `https://huggingface.co/${repo}` }];
}
