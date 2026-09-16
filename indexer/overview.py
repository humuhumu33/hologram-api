"""Model passports: a standardised overview of every indexed model, derived from verified files and Hugging Face.

For each model:
  - README.md and config.json are fetched at the indexed revision and checked against their SHA-256 addresses in
    the index before they are read (a mismatch writes nothing);
  - one Hugging Face API call supplies structured fields (evaluation results, family, providers, papers, Spaces);
  - every value is stored as {"value", "source"} so each fact traces to `api:<field>`, `file:<path>@sha256:<hex>`
    or `derived:<rule>`. Fields without a trustworthy source are omitted, never guessed.

    python indexer/overview.py [--repos owner/name ...] [--extra-list path] [--workers 4]

Writes v1/overview/huggingface.co/{owner}/{repo}.json (format hologram.overview/v1) and v1/overview/index.json.
Models listed in --extra-list that are not indexed use Hugging Face `main`, with provenance saying so.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.parse

import requests

HF = "https://huggingface.co"
UA = {"User-Agent": "hologram-index/1 (+https://github.com/humuhumu33/hologram-api)"}
if os.environ.get("HF_TOKEN"):
    UA["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
EXPAND = ["author", "cardData", "config", "createdAt", "lastModified", "tags", "pipeline_tag", "library_name",
          "safetensors", "gguf", "spaces", "transformersInfo", "evalResults", "inferenceProviderMapping",
          "baseModels", "childrenModelCount", "downloadsAllTime", "likes", "downloads", "gated", "sha"]

# ---- standardisation tables ---------------------------------------------------------------------------------------

TASKS = {
    "text-generation": ("Text generation", ["text"], ["text"]),
    "text2text-generation": ("Text to text", ["text"], ["text"]),
    "image-text-to-text": ("Vision language", ["image", "text"], ["text"]),
    "video-text-to-text": ("Video language", ["video", "text"], ["text"]),
    "any-to-any": ("Any to any", ["text", "image", "audio"], ["text", "image", "audio"]),
    "audio-text-to-text": ("Audio language", ["audio", "text"], ["text"]),
    "text-to-image": ("Text to image", ["text"], ["image"]),
    "image-to-image": ("Image to image", ["image"], ["image"]),
    "text-to-video": ("Text to video", ["text"], ["video"]),
    "image-to-video": ("Image to video", ["image"], ["video"]),
    "text-to-speech": ("Text to speech", ["text"], ["audio"]),
    "text-to-audio": ("Text to audio", ["text"], ["audio"]),
    "automatic-speech-recognition": ("Speech recognition", ["audio"], ["text"]),
    "audio-to-audio": ("Audio to audio", ["audio"], ["audio"]),
    "audio-classification": ("Audio classification", ["audio"], ["label"]),
    "feature-extraction": ("Embeddings", ["text"], ["vector"]),
    "sentence-similarity": ("Sentence similarity", ["text"], ["vector"]),
    "fill-mask": ("Fill mask", ["text"], ["text"]),
    "text-classification": ("Text classification", ["text"], ["label"]),
    "token-classification": ("Token classification", ["text"], ["label"]),
    "zero-shot-classification": ("Zero shot classification", ["text"], ["label"]),
    "question-answering": ("Question answering", ["text"], ["text"]),
    "summarization": ("Summarization", ["text"], ["text"]),
    "translation": ("Translation", ["text"], ["text"]),
    "image-classification": ("Image classification", ["image"], ["label"]),
    "object-detection": ("Object detection", ["image"], ["boxes"]),
    "image-segmentation": ("Image segmentation", ["image"], ["mask"]),
    "depth-estimation": ("Depth estimation", ["image"], ["depth"]),
    "zero-shot-image-classification": ("Zero shot image classification", ["image", "text"], ["label"]),
    "image-to-text": ("Image to text", ["image"], ["text"]),
    "time-series-forecasting": ("Time series forecasting", ["series"], ["series"]),
    "reinforcement-learning": ("Reinforcement learning", ["state"], ["action"]),
}

# SPDX id → (display name, commercial use: yes / no / conditions, note)
LICENSES = {
    "apache-2.0": ("Apache 2.0", "yes", "Permissive, with patent grant"),
    "mit": ("MIT", "yes", "Permissive"),
    "bsd-2-clause": ("BSD 2 Clause", "yes", "Permissive"),
    "bsd-3-clause": ("BSD 3 Clause", "yes", "Permissive"),
    "cc-by-4.0": ("CC BY 4.0", "yes", "Attribution required"),
    "cc-by-sa-4.0": ("CC BY SA 4.0", "yes", "Attribution, share alike"),
    "cc-by-nc-4.0": ("CC BY NC 4.0", "no", "Non commercial only"),
    "cc-by-nc-sa-4.0": ("CC BY NC SA 4.0", "no", "Non commercial, share alike"),
    "cc-by-nc-nd-4.0": ("CC BY NC ND 4.0", "no", "Non commercial, no derivatives"),
    "gpl-3.0": ("GPL 3.0", "yes", "Copyleft"),
    "agpl-3.0": ("AGPL 3.0", "yes", "Network copyleft"),
    "lgpl-3.0": ("LGPL 3.0", "yes", "Weak copyleft"),
    "openrail": ("OpenRAIL", "conditions", "Use restrictions apply"),
    "openrail++": ("OpenRAIL++", "conditions", "Use restrictions apply"),
    "creativeml-openrail-m": ("CreativeML OpenRAIL M", "conditions", "Use restrictions apply"),
    "bigscience-openrail-m": ("BigScience OpenRAIL M", "conditions", "Use restrictions apply"),
    "llama3": ("Llama 3 Community", "conditions", "Conditions above 700M monthly users"),
    "llama3.1": ("Llama 3.1 Community", "conditions", "Conditions above 700M monthly users"),
    "llama3.2": ("Llama 3.2 Community", "conditions", "Conditions above 700M monthly users"),
    "llama3.3": ("Llama 3.3 Community", "conditions", "Conditions above 700M monthly users"),
    "llama4": ("Llama 4 Community", "conditions", "Conditions above 700M monthly users"),
    "gemma": ("Gemma Terms", "conditions", "Prohibited use policy applies"),
    "openmdw-1.0": ("OpenMDW 1.0", "yes", "Permissive, for models"),
    "openmdw-1.1": ("OpenMDW 1.1", "yes", "Permissive, for models"),
}

RELATIONS = {"finetune": "Fine tuned from", "quantized": "Quantized from", "adapter": "Adapter for", "merge": "Merged from"}

CONFIG_KEYS = {
    "layers": ["num_hidden_layers", "n_layer", "num_layers", "n_layers"],
    "hidden": ["hidden_size", "n_embd", "d_model", "dim"],
    "heads": ["num_attention_heads", "n_head", "num_heads", "n_heads"],
    "kv_heads": ["num_key_value_heads", "multi_query_group_num", "n_kv_heads"],
    "vocab": ["vocab_size", "n_vocab"],
    "context": ["max_position_embeddings", "max_sequence_length", "seq_length", "n_positions", "n_ctx"],
    "experts": ["num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"],
    "experts_active": ["num_experts_per_tok", "moe_top_k", "num_experts_per_token", "top_k_experts"],
    "rope_theta": ["rope_theta"],
    "tied": ["tie_word_embeddings"],
}

DTYPES = {"BF16": "BF16", "F16": "FP16", "F32": "FP32", "F8_E4M3": "FP8", "F8_E5M2": "FP8", "I8": "INT8",
          "U8": "UINT8", "I32": "INT32", "U32": "UINT32", "I64": "INT64", "BOOL": "BOOL", "F64": "FP64", "U4": "INT4",
          "I4": "INT4", "F4": "FP4"}

# ---- rate aware HTTP ----------------------------------------------------------------------------------------------

_lock = threading.Lock()
_pause_until = 0.0


def fetch(url, kind="json", tries=5):
    global _pause_until
    for attempt in range(tries):
        wait = _pause_until - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, headers=UA, timeout=60)
        except requests.RequestException:
            time.sleep(2 * 2 ** attempt)
            continue
        limit = re.search(r"r=(\d+);t=(\d+)", r.headers.get("RateLimit", ""))
        if limit and int(limit.group(1)) < 8:
            with _lock:
                _pause_until = max(_pause_until, time.time() + int(limit.group(2)) + 1)
        if r.status_code == 429:
            with _lock:
                _pause_until = max(_pause_until, time.time() + (int(limit.group(2)) + 1 if limit else 30))
            continue
        if r.status_code in (401, 403, 404):
            return None
        if r.ok:
            return r.json() if kind == "json" else r.content
        time.sleep(2 * 2 ** attempt)
    return None


# ---- helpers ------------------------------------------------------------------------------------------------------

def field(value, source):
    return {"value": value, "source": source} if value not in (None, "", [], {}) else None


def clean(d):
    """Drop omitted fields recursively."""
    if isinstance(d, dict):
        out = {k: clean(v) for k, v in d.items()}
        return {k: v for k, v in out.items() if v not in (None, {}, [])}
    if isinstance(d, list):
        return [x for x in (clean(v) for v in d) if x not in (None, {}, [])]
    return d


def first(cfg, keys):
    for k in keys:
        if isinstance(cfg.get(k), (int, float, bool)) and not isinstance(cfg.get(k), bool) or (k == "tie_word_embeddings" and isinstance(cfg.get(k), bool)):
            return k, cfg[k]
    return None, None


FRONT = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
DENY = re.compile(r"\b(api service|cloud|coming soon|stay tuned|sign up|pricing|discord|join our|star us|wechat)\b", re.I)


def extract_summary(readme):
    """First descriptive paragraph, cleaned, two sentences at most. Returns None when nothing qualifies."""
    text = FRONT.sub("", readme)
    text = re.sub(r"```.*?```", "\n\n", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b or b.startswith(("#", ">", "|", "<", "- ", "* ", "+ ", "!", "[!", "1.", "```")) or b.startswith("**") and b.rstrip().endswith("**") and len(b.split()) < 8:
            continue
        if re.fullmatch(r"(\[?!\[[^\]]*\]\([^)]*\)\]?(\([^)]*\))?\s*)+", b):
            continue
        plain = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", b)
        plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", plain)
        plain = re.sub(r"<[^>]+>", "", plain)
        plain = re.sub(r"[*_`~]+", "", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        if len(plain.split()) < 12 or DENY.search(plain):
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", plain)
        out = ""
        for s in sentences[:2]:
            if len(out) + len(s) + 1 > 320:
                break
            out = f"{out} {s}".strip()
        if not out:
            out = plain[:317].rsplit(" ", 1)[0] + "…"
        return out
    return None


def fmt_params(n):
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            v = n / div
            return f"{v:.1f}".rstrip("0").rstrip(".") + unit if v < 100 else f"{round(v)}{unit}"
    return str(n)


BRANDS = {"deepseek": "DeepSeek", "qwen": "Qwen", "llama": "Llama", "gemma": "Gemma", "mistral": "Mistral",
          "mixtral": "Mixtral", "phi": "Phi", "glm": "GLM", "chatglm": "ChatGLM", "minicpm": "MiniCPM", "internvl": "InternVL",
          "internlm": "InternLM", "bert": "BERT", "roberta": "RoBERTa", "xlm": "XLM", "t5": "T5", "mt5": "mT5", "gpt": "GPT",
          "gpt2": "GPT 2", "gptj": "GPT J", "whisper": "Whisper", "clip": "CLIP", "siglip": "SigLIP", "llava": "LLaVA",
          "smollm": "SmolLM", "olmo": "OLMo", "olmoe": "OLMoE", "falcon": "Falcon", "mamba": "Mamba", "jamba": "Jamba",
          "granite": "Granite", "exaone": "EXAONE", "ernie": "ERNIE", "hunyuan": "Hunyuan", "kimi": "Kimi", "minimax": "MiniMax",
          "stablelm": "StableLM", "moe": "MoE", "vl": "VL", "tts": "TTS", "asr": "ASR", "mlp": "MLP", "dit": "DiT", "vit": "ViT"}


def family_name(model_type):
    """model_type → display name: 'qwen3_5' → 'Qwen 3.5', 'deepseek_v41' → 'DeepSeek V41', 'bert' → 'BERT'."""
    if not model_type:
        return None
    words = []
    for token in re.split(r"[_\-\s]+", str(model_type)):
        m = re.fullmatch(r"([a-z]+?)(\d+)", token)
        if token in BRANDS:
            words.append(BRANDS[token])
        elif re.fullmatch(r"v\d+", token):
            words.append(token.upper())
        elif m and m.group(1) in BRANDS:
            words += [BRANDS[m.group(1)], m.group(2)]
        elif re.fullmatch(r"\d+", token) and words and re.fullmatch(r"\d+", words[-1]):
            words[-1] = f"{words[-1]}.{token}"
        else:
            words.append(token[:1].upper() + token[1:])
    return " ".join(words)


# ---- one model ----------------------------------------------------------------------------------------------------

def build(root, repo, indexed, paper_cache):
    doc = None
    latest = os.path.join(root, "huggingface.co", *repo.split("/"), "latest.json")
    if indexed and os.path.exists(latest):
        doc = json.load(open(latest, encoding="utf-8"))
    query = "&".join(f"expand[]={e}" for e in EXPAND)
    api = fetch(f"{HF}/api/models/{repo}?{query}")
    if not api:
        return None
    revision = doc["revision"] if doc else api.get("sha")
    files = {f["path"]: f for f in (doc or {}).get("files", [])}

    def verified_file(path):
        f = files.get(path)
        url = f["url"] if f else f"{HF}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"
        body = fetch(url, kind="bytes")
        if body is None:
            return None, None
        digest = hashlib.sha256(body).hexdigest()
        if f and f"sha256:{digest}" != f["address"]:
            raise ValueError(f"{repo}/{path}: bytes do not match the index address")
        return body, f"file:{path}@sha256:{digest}"

    readme_bytes, readme_src = verified_file("README.md")
    config_bytes, config_src = verified_file("config.json")
    readme = readme_bytes.decode("utf-8", "replace") if readme_bytes else ""
    try:
        config = json.loads(config_bytes) if config_bytes else {}
    except ValueError:
        config = {}
    cfg = config.get("text_config") or config.get("llm_config") or config.get("language_config") or config
    card = api.get("cardData") or {}
    tags = api.get("tags") or []
    task_id = api.get("pipeline_tag") or card.get("pipeline_tag")
    task = TASKS.get(task_id)
    params = (api.get("safetensors") or {}).get("total") or (api.get("gguf") or {}).get("total")
    license_id = card.get("license") or next((t[8:] for t in tags if t.startswith("license:")), None)
    lic = LICENSES.get(str(license_id).lower()) if license_id else None
    model_type = config.get("model_type") or (api.get("config") or {}).get("model_type") or (api.get("gguf") or {}).get("architecture")
    languages = card.get("language")
    if isinstance(languages, str):
        languages = [languages]

    # summary: extracted from the verified README, else a deterministic template from structured fields
    summary = extract_summary(readme) if readme else None
    if summary:
        summary_field = field(summary, readme_src)
    else:
        name = repo.split("/")[1]
        bits = [f"{name} is a"]
        if params:
            bits.append(fmt_params(params))
        if family_name(model_type):
            bits.append(family_name(model_type))
        bits.append("model")
        if task:
            bits.append(f"for {task[0].lower()}")
        bits.append(f"by {repo.split('/')[0]}")
        sentence = " ".join(bits)
        if api.get("createdAt"):
            sentence += f", released {time.strftime('%b %Y', time.strptime(api['createdAt'][:10], '%Y-%m-%d'))}"
        if lic:
            sentence += f", under {lic[0]}"
        summary_field = field(sentence + ".", "derived:template")

    # parameters by dtype
    precision = None
    by_dtype = (api.get("safetensors") or {}).get("parameters") or {}
    if by_dtype and params:
        merged = {}
        for k, v in by_dtype.items():
            merged[DTYPES.get(k, k)] = merged.get(DTYPES.get(k, k), 0) + v
        parts = sorted(merged.items(), key=lambda kv: -kv[1])
        precision = ", ".join(f"{k} {round(100 * v / params)}%" if len(parts) > 1 else k for k, v in parts if round(100 * v / params) > 0 or len(parts) == 1)

    arch = {}
    for concept, keys in CONFIG_KEYS.items():
        key, value = first(cfg, keys)
        if key is None and cfg is not config:
            key, value = first(config, keys)
        if key is not None:
            arch[concept] = field(value, f"{config_src}#{key}")
    context = arch.get("context") or field((api.get("gguf") or {}).get("context_length"), "api:gguf.context_length")
    quant = config.get("quantization_config") or cfg.get("quantization_config")
    if isinstance(quant, dict):
        method = quant.get("quant_method") or quant.get("quantization_method")
        bits = quant.get("bits") or quant.get("weight_bits")
        arch["quantization"] = field(" ".join(str(x) for x in (method, f"{bits} bit" if bits else None) if x), f"{config_src}#quantization_config")
    for key, label in (("vision_config", "vision"), ("audio_config", "audio")):
        if isinstance(config.get(key), dict):
            arch[f"{label}_encoder"] = field(config[key].get("model_type") or "present", f"{config_src}#{key}")

    # benchmarks: structured only
    benchmarks = []
    for e in api.get("evalResults") or []:
        data = e.get("data") or {}
        ds = data.get("dataset") or {}
        if data.get("value") is None or not ds.get("id"):
            continue
        benchmarks.append({"dataset": ds.get("id"), "task": ds.get("task_id"), "value": data.get("value"),
                           "date": data.get("date"), "verified": bool(e.get("verified")),
                           "source": (data.get("source") or {}).get("url"), "notes": data.get("notes")})
    if not benchmarks:
        for entry in card.get("model-index") or []:
            for result in entry.get("results") or []:
                for metric in result.get("metrics") or []:
                    if metric.get("value") is not None:
                        benchmarks.append({"dataset": (result.get("dataset") or {}).get("name") or (result.get("dataset") or {}).get("type"),
                                           "task": metric.get("name") or metric.get("type"), "value": metric.get("value"),
                                           "verified": bool(metric.get("verified")), "source": "model card"})

    base = api.get("baseModels") or {}
    papers = []
    for arxiv in [t[6:] for t in tags if t.startswith("arxiv:")][:6]:
        if arxiv not in paper_cache:
            p = fetch(f"{HF}/api/papers/{arxiv}")
            paper_cache[arxiv] = {"id": arxiv, "title": re.sub(r"\s+", " ", p["title"]).strip(),
                                  "first_author": ((p.get("authors") or [{}])[0] or {}).get("name"),
                                  "authors": len(p.get("authors") or []),
                                  "year": (p.get("publishedAt") or "")[:4]} if p and p.get("title") else {"id": arxiv}
        papers.append(paper_cache[arxiv])
    datasets = card.get("datasets")
    if isinstance(datasets, str):
        datasets = [datasets]
    providers = sorted(
        ({"provider": k, "status": v.get("status"), "task": v.get("task")} for k, v in (api.get("inferenceProviderMapping") or {}).items()
         if isinstance(v, dict) and v.get("status") == "live"), key=lambda p: p["provider"])
    info = api.get("transformersInfo") or {}

    result = {
        "format": "hologram.overview/v1",
        "name": repo, "revision": revision, "manifest": (doc or {}).get("manifest"),
        "addressed": bool(doc), "generated": time.strftime("%Y-%m-%d", time.gmtime()),
        "provenance": {"api": f"{HF}/api/models/{repo}", "readme": readme_src, "config": config_src},
        "summary": summary_field,
        "glance": {
            "task": field(task[0] if task else task_id, "api:pipeline_tag"),
            "input": field(task[1] if task else None, "derived:pipeline_tag"),
            "output": field(task[2] if task else None, "derived:pipeline_tag"),
            "parameters": field(params, "api:safetensors.total" if (api.get("safetensors") or {}).get("total") else "api:gguf.total"),
            "experts": arch.get("experts"), "experts_active": arch.get("experts_active"),
            "architecture": field(family_name(model_type), f"{config_src}#model_type" if config.get("model_type") else "api:config.model_type"),
            "architecture_id": field(model_type, f"{config_src}#model_type" if config.get("model_type") else "api:config.model_type"),
            "context": context,
            "precision": field(precision, "api:safetensors.parameters"),
            "format": field("GGUF" if api.get("gguf") else "Safetensors" if api.get("safetensors") else None, "api:safetensors|gguf"),
            "library": field(api.get("library_name") or card.get("library_name"), "api:library_name"),
            "license": field({"id": license_id, "name": lic[0] if lic else license_id, "commercial": lic[1] if lic else None,
                              "note": lic[2] if lic else None} if license_id else None, "api:cardData.license"),
            "languages": field(languages, "api:cardData.language"),
            "base": field({"relation": RELATIONS.get(base.get("relation"), base.get("relation")),
                           "models": [m.get("id") for m in base.get("models") or []]} if base.get("models") else None, "api:baseModels"),
            "released": field((api.get("createdAt") or "")[:10], "api:createdAt"),
            "updated": field((api.get("lastModified") or "")[:10], "api:lastModified"),
            "likes": field(api.get("likes"), "api:likes"),
            "downloads_all_time": field(api.get("downloadsAllTime"), "api:downloadsAllTime"),
        },
        "architecture": arch,
        "benchmarks": field(benchmarks, "api:evalResults" if api.get("evalResults") else "api:cardData.model-index"),
        "family": {
            "children": field({RELATIONS.get(k, k).replace(" from", "").replace(" for", ""): v
                               for k, v in (api.get("childrenModelCount") or {}).items() if v}, "api:childrenModelCount"),
            "children_raw": field({k: v for k, v in (api.get("childrenModelCount") or {}).items() if v}, "api:childrenModelCount"),
        },
        "run": {
            "auto_model": field(info.get("auto_model"), "api:transformersInfo.auto_model"),
            "processor": field(info.get("processor"), "api:transformersInfo.processor"),
            "providers": field(providers, "api:inferenceProviderMapping"),
        },
        "papers": field(papers, "api:tags(arxiv)+api:papers"),
        "doi": field(next((t[4:] for t in tags if t.startswith("doi:")), None), "api:tags(doi)"),
        "datasets": field(datasets, "api:cardData.datasets"),
        "spaces": field({"count": len(api.get("spaces") or []), "top": (api.get("spaces") or [])[:3]}
                        if api.get("spaces") else None, "api:spaces"),
    }
    return clean(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--repos", nargs="*")
    parser.add_argument("--extra-list", help="file with extra repo ids (one per line), e.g. the site's trending list")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="rebuild even when the revision is unchanged")
    args = parser.parse_args()
    root = os.path.join(args.out, "v1")
    index = {m["name"]: m for m in json.load(open(os.path.join(root, "index.json"), encoding="utf-8"))["models"]}
    repos = list(args.repos or index)
    if args.extra_list and os.path.exists(args.extra_list):
        repos += [r.strip() for r in open(args.extra_list, encoding="utf-8") if r.strip() and r.strip() not in repos]
    out_index_path = os.path.join(root, "overview", "index.json")
    previous = json.load(open(out_index_path, encoding="utf-8"))["models"] if os.path.exists(out_index_path) else {}
    summary, paper_cache, started = dict(previous), {}, time.time()
    counts = {"summary_extracted": 0, "summary_template": 0, "context": 0, "benchmarks": 0, "parameters": 0, "refused": 0, "missing": 0}

    def one(repo):
        rev = index.get(repo, {}).get("revision")
        prev = previous.get(repo)
        if not args.force and prev and rev and prev.get("revision") == rev and prev.get("generated", "") >= time.strftime("%Y-%m-%d", time.gmtime(time.time() - 7 * 86400)):
            return repo, "unchanged", None
        try:
            return repo, "ok", build(root, repo, repo in index, paper_cache)
        except ValueError as error:
            return repo, f"refused: {error}", None

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for n, (repo, status, doc) in enumerate(pool.map(one, repos), 1):
            if status.startswith("refused"):
                counts["refused"] += 1
                print(status)
                continue
            if status == "unchanged":
                continue
            if not doc:
                counts["missing"] += 1
                continue
            target = os.path.join(root, "overview", "huggingface.co", *repo.split("/")) + ".json"
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
            summary[repo] = {"revision": doc.get("revision"), "generated": doc["generated"], "addressed": doc["addressed"],
                             "summary": (doc.get("summary") or {}).get("source", "").split(":")[0] or None}
            if n % 100 == 0:
                print(f"{n}/{len(repos)} {(time.time() - started) / 60:.1f} min")
    for repo, s in summary.items():
        counts["summary_extracted" if s.get("summary") == "file" else "summary_template"] += 1
    with open(out_index_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"format": "hologram.overview-index/v1", "models": dict(sorted(summary.items()))}, indent=2) + "\n")
    total = len(summary) or 1
    print(f"{len(summary)} overviews; summaries extracted {100 * counts['summary_extracted'] // total}%, "
          f"template {100 * counts['summary_template'] // total}%; refused {counts['refused']}, missing {counts['missing']}; "
          f"{(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
