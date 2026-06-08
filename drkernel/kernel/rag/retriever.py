import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_:+.-]+")
ARCHITECTURE_RE = re.compile(
    r"You are given the following architecture:\s*```(?:python)?\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)
K1 = 1.5
B = 0.75
OPERATOR_HINTS = [
    "matmul",
    "gemm",
    "softmax",
    "layernorm",
    "conv1d",
    "conv2d",
    "conv3d",
    "reduction",
    "attention",
    "dropout",
    "flash",
    "shared memory",
    "coalescing",
    "tensor core",
    "block pointer",
    "make_block_ptr",
    "warp",
    "occupancy",
]
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "architecture",
    "as",
    "class",
    "code",
    "custom",
    "def",
    "example",
    "for",
    "functional",
    "given",
    "import",
    "in",
    "input",
    "kernels",
    "model",
    "new",
    "nn",
    "not",
    "operators",
    "optimize",
    "output",
    "outputs",
    "please",
    "pseudocode",
    "real",
    "replace",
    "return",
    "self",
    "step",
    "super",
    "the",
    "think",
    "to",
    "torch",
    "triton",
    "with",
    "write",
    "you",
}
QUERY_MODES = {"full_prompt", "architecture_code", "op_signature"}
OPERATOR_SIGNATURE_PATTERNS = [
    (re.compile(r"\bconv1d\b"), ["conv1d", "convolution"]),
    (re.compile(r"\bconv2d\b"), ["conv2d", "convolution"]),
    (re.compile(r"\bconv3d\b"), ["conv3d", "convolution"]),
    (re.compile(r"layernorm"), ["layernorm", "normalization"]),
    (re.compile(r"dropout"), ["dropout", "mask"]),
    (re.compile(r"softmax"), ["softmax", "reduction"]),
    (re.compile(r"\bmatmul\b|\bmm\b|\bbmm\b|@"), ["matmul", "gemm"]),
    (re.compile(r"\bgelu\b"), ["gelu", "activation"]),
    (re.compile(r"\brelu\b"), ["relu", "activation"]),
    (re.compile(r"\bsilu\b|\bswish\b"), ["silu", "activation"]),
    (re.compile(r"\bsigmoid\b"), ["sigmoid", "activation"]),
    (re.compile(r"\bclamp\b"), ["clamp", "activation"]),
    (re.compile(r"\badd\b|\+"), ["add", "elementwise"]),
    (re.compile(r"\bmul\b|\*"), ["mul", "elementwise"]),
    (re.compile(r"\bdiv\b|/"), ["div", "elementwise"]),
    (re.compile(r"\bsub\b|-"), ["sub", "elementwise"]),
    (re.compile(r"\bsum\b"), ["sum", "reduction"]),
    (re.compile(r"\bmean\b"), ["mean", "reduction"]),
    (re.compile(r"\bmax\b"), ["max", "reduction"]),
    (re.compile(r"\bmin\b"), ["min", "reduction"]),
    (re.compile(r"\bbias\b"), ["bias"]),
]


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def extract_architecture_code(prompt: str) -> str | None:
    match = ARCHITECTURE_RE.search(prompt)
    if not match:
        return None
    architecture = match.group(1).strip()
    return architecture or None


def extract_operator_signature(arch_code: str) -> str:
    lowered = arch_code.lower()
    signature_terms: list[str] = []
    seen = set()

    for pattern, terms in OPERATOR_SIGNATURE_PATTERNS:
        if not pattern.search(lowered):
            continue
        for term in terms:
            if term in seen:
                continue
            signature_terms.append(term)
            seen.add(term)

    return " ".join(signature_terms)


def build_retrieval_query(prompt: str, mode: str) -> str:
    normalized_mode = (mode or "op_signature").lower()
    if normalized_mode not in QUERY_MODES:
        normalized_mode = "op_signature"

    architecture = extract_architecture_code(prompt)

    if normalized_mode == "full_prompt":
        return prompt

    if architecture is None:
        return prompt

    if normalized_mode == "architecture_code":
        return architecture

    signature = extract_operator_signature(architecture)
    if signature:
        return signature
    return architecture


def expand_query(text: str) -> str:
    lowered = text.lower()
    extra_terms = [term for term in OPERATOR_HINTS if term in lowered]
    return text + "\n" + " ".join(extra_terms) if extra_terms else text


def tokenize_query(text: str, dedupe: bool = True) -> list[str]:
    raw_tokens = tokenize(text)
    filtered_tokens: list[str] = []
    seen = set()

    for token in raw_tokens:
        if token in QUERY_STOPWORDS:
            continue
        if len(token) == 1:
            continue
        if token.isdigit():
            continue
        if dedupe:
            if token in seen:
                continue
            seen.add(token)
        filtered_tokens.append(token)

    return filtered_tokens or raw_tokens


class BM25ManualRetriever:
    def __init__(
        self,
        index_path: str | Path,
        query_mode: str = "op_signature",
        max_chunks_per_source: int | None = 1,
        dedupe_query_tokens: bool = True,
    ):
        index = json.loads(Path(index_path).read_text())
        self.avgdl = float(index["avgdl"])
        self.doc_lens = index["doc_lens"]
        self.idf = index["idf"]
        self.doc_tfs = [Counter(doc_tf) for doc_tf in index["doc_tfs"]]
        self.chunks = index["chunks"]
        self.query_mode = query_mode
        self.max_chunks_per_source = max_chunks_per_source
        self.dedupe_query_tokens = dedupe_query_tokens

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        tf = self.doc_tfs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            idf = self.idf.get(term)
            if idf is None:
                continue
            denom = freq + K1 * (1 - B + B * doc_len / max(self.avgdl, 1e-6))
            score += idf * (freq * (K1 + 1)) / denom
        return score

    @lru_cache(maxsize=4096)
    def retrieve(self, prompt: str, topk: int = 4) -> list[dict]:
        query_text = build_retrieval_query(prompt, mode=self.query_mode)
        expanded = expand_query(query_text)
        query_tokens = tokenize_query(expanded, dedupe=self.dedupe_query_tokens)
        if not query_tokens:
            return []

        scored = []
        for idx in range(len(self.chunks)):
            score = self._score(query_tokens, idx)
            if score > 0:
                scored.append((score, idx))
        scored.sort(reverse=True)

        if not self.max_chunks_per_source or self.max_chunks_per_source < 1:
            return [self.chunks[idx] for _, idx in scored[:topk]]

        selected = []
        source_counts = Counter()
        for _, idx in scored:
            chunk = self.chunks[idx]
            source_id = chunk.get("source_id")
            if source_counts[source_id] >= self.max_chunks_per_source:
                continue
            selected.append(chunk)
            source_counts[source_id] += 1
            if len(selected) >= topk:
                break
        return selected


def format_retrieved_context(chunks: list[dict], max_chars: int = 4000) -> str:
    if not chunks:
        return ""

    sections = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or chunk.get("source_id") or f"Doc {i}"
        text = chunk["text"].strip()
        section = f"[Doc {i}] {title}\n{text}"
        sections.append(section)

    merged = "\n\n".join(sections)
    if len(merged) <= max_chars:
        return merged
    return merged[:max_chars].rsplit(" ", 1)[0]
