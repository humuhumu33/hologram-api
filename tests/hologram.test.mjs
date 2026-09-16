// node --test tests/   — offline tests always run; network tests run when HOLOGRAM_NETWORK=1.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { address, canonicalManifest, check, fetchVerified, parseAddress, Refused, resolve, verify } from "../hologram.js";

const NETWORK = process.env.HOLOGRAM_NETWORK === "1";

test("known vectors match the Rust server and Python client", async () => {
  const empty = await address(new Uint8Array(0), ["sha256", "blake3", "gitsha1"]);
  assert.equal(empty.sha256, "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert.equal(empty.blake3, "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
  assert.equal(empty.gitsha1, "gitsha1:e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
  assert.equal(empty.size, 0);
});

test("strings, bytes, blobs and streams address identically", async () => {
  const text = "identical payload";
  const bytes = new TextEncoder().encode(text);
  const expected = await address(text);
  assert.deepEqual(await address(bytes), expected);
  assert.deepEqual(await address(new Blob([bytes])), expected);
  assert.deepEqual(await address(new Blob([bytes.slice(0, 5), bytes.slice(5)]).stream()), expected);
  assert.deepEqual(await address(new Response(bytes)), expected);
  assert.equal(expected.size, bytes.byteLength);
});

test("one flipped bit is a different address, and verify says so without throwing", async () => {
  const good = new TextEncoder().encode("identical payload");
  const bad = good.slice(); bad[3] ^= 1;
  const { sha256 } = await address(good);
  assert.equal((await verify(good, sha256)).ok, true);
  const result = await verify(bad, sha256);
  assert.equal(result.ok, false);
  assert.equal(result.expected, sha256);
  await assert.rejects(check(bad, sha256), (e) => e instanceof Refused && e.code === "ADDRESS_MISMATCH");
});

test("addresses are parsed strictly", () => {
  assert.equal(parseAddress("SHA256:" + "A".repeat(64)).digest, "a".repeat(64));
  for (const bad of ["sha256:abc", "md5:d41d8cd98f00b204e9800998ecf8427e", "holo://blake3:00", "", null]) {
    assert.throws(() => parseAddress(bad), (e) => e.code === "BAD_ADDRESS");
  }
});

test("canonical manifest of a published document reproduces its address", async () => {
  const doc = JSON.parse(readFileSync(new URL("../v1/huggingface.co/hf-internal-testing/tiny-random-gpt2/latest.json", import.meta.url)));
  const { blake3 } = await address(canonicalManifest(doc), ["blake3"]);
  assert.equal(blake3, doc.manifest);
  // The same address hologram-api-server's Rust resolver produced independently.
  assert.equal(blake3, "blake3:84f5556b153b70dd120468684ff501e3bb4905a251ebff3b32e5bd8989eee76b");
});

test("fetchVerified returns honest bytes and refuses a tampering server before returning any", async () => {
  const payload = new TextEncoder().encode("weights ".repeat(50000));
  const tampered = payload.slice(); tampered[123456] ^= 0x80;
  const server = createServer((req, res) => res.end(req.url === "/honest" ? payload : tampered));
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const { sha256 } = await address(payload);
    const bytes = await fetchVerified(`${base}/honest`, sha256);
    assert.equal(bytes.byteLength, payload.byteLength);
    await assert.rejects(fetchVerified(`${base}/tampered`, sha256), (e) => e.code === "ADDRESS_MISMATCH" && e.expected === sha256);
  } finally {
    server.close();
  }
});

test("resolve refuses names that are not repositories", async () => {
  await assert.rejects(resolve("../etc/passwd"), (e) => e.code === "NOT_FOUND");
});

test("network: resolve from the public hosts, then fetch a real weight file verified", { skip: !NETWORK }, async () => {
  const doc = await resolve("hf-internal-testing/tiny-random-gpt2");
  const file = doc.files.find((f) => f.path === "model.safetensors");
  const bytes = await fetchVerified(file.url, file.address);
  assert.equal(bytes.byteLength, file.size);
  await assert.rejects(resolve("hf-internal-testing/tiny-random-gpt2", { manifest: "blake3:" + "0".repeat(64) }), (e) => e.code === "ADDRESS_MISMATCH");
});
