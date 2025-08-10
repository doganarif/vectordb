#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import pathlib
import re
import shutil
import sys
import time
from typing import Iterable, List, Tuple

import requests

# Ensure project root is on sys.path so `sdk` can be imported when run directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


log = logging.getLogger("e2e")


def await_health(base_url: str, timeout_seconds: int = 15) -> None:
    deadline = time.time() + timeout_seconds
    last_err = None
    url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                log.info("API is healthy at %s", url)
                return
            last_err = f"HTTP {r.status_code}: {r.text}"
        except Exception as e:  # noqa: BLE001 - simple e2e script
            last_err = str(e)
        time.sleep(0.3)
    raise RuntimeError(f"Health check failed for {base_url}: {last_err}")


# ───────────────────────────────── CLI formatting helpers ─────────────────────────────────
class Palette:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def color_enabled(no_color_flag: bool) -> bool:
    return sys.stdout.isatty() and not no_color_flag


def c(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{code}{text}{Palette.RESET}"


def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def section(title: str, icon: str, enabled: bool) -> None:
    width = max(40, term_width() - 2)
    line = "═" * width
    header = f" {icon}  {title} " if icon else f" {title} "
    pad = max(0, width - len(header))
    bar = header + ("═" * pad)
    print(c(line, Palette.CYAN + Palette.BOLD, enabled))
    print(c(bar, Palette.CYAN + Palette.BOLD, enabled))
    print(c(line, Palette.CYAN + Palette.BOLD, enabled))


def kv(
    key: str,
    value: str,
    enabled: bool,
    key_color: str = Palette.MAGENTA,
    value_color: str = Palette.WHITE,
) -> None:
    print(
        f"  {c(key+':', key_color + Palette.BOLD, enabled)} {c(value, value_color, enabled)}"
    )


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def hashed_bow_embedding(text: str, dim: int = 64) -> List[float]:
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:8], "big") % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def pretty_results(items: Iterable[dict], enabled: bool = False) -> List[str]:
    lines: List[str] = []
    for i, item in enumerate(items, start=1):
        score = item.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (float, int)) else str(score)
        # color score: green for positive, yellow near 0, red for negative
        score_val = float(score) if isinstance(score, (float, int)) else None
        score_col = Palette.WHITE
        if score_val is not None:
            if score_val > 0.4:
                score_col = Palette.GREEN
            elif score_val > 0:
                score_col = Palette.YELLOW
            else:
                score_col = Palette.RED
        lines.append(
            f"{c(str(i).rjust(2), Palette.BLUE+Palette.BOLD, enabled)}. "
            f"score={c(score_str, score_col+Palette.BOLD, enabled)} "
            f"chunk_id={c(item.get('chunk_id', ''), Palette.WHITE, enabled)} "
            f"doc={c(item.get('document_id', ''), Palette.DIM, enabled)}"
        )
        text = item.get("text", "").strip().replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        lines.append(f"    {c('text=', Palette.DIM, enabled)}{text}")
    return lines


def timeit(label: str) -> Tuple[str, float]:
    return label, time.perf_counter()


def elapsed(start: Tuple[str, float]) -> str:
    label, t0 = start
    return f"{label} in {(time.perf_counter() - t0)*1000:.1f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end demo for Vector DB")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://127.0.0.1:8000"),
        help="API base URL",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=int(os.getenv("EMBED_DIM", "64")),
        help="Fallback embedding dimension when not using Cohere",
    )
    parser.add_argument(
        "--no-cohere",
        action="store_true",
        help="Force local fallback embeddings even if COHERE_API_KEY is set",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete created resources at the end of the demo",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    base_url = args.base_url
    log.info("Using base_url=%s", base_url)
    color_on = color_enabled(args.no_color)

    # Wait for API to be available if it's starting up
    section("Setup and health check", "🚀", color_on)
    try:
        await_health(base_url)
    except Exception as e:  # noqa: BLE001 - simple e2e script
        raise SystemExit(f"API is not healthy: {e}")

    from sdk.client import VectorDBClient

    client = VectorDBClient(base_url=base_url)

    # Decide embedding strategy
    cohere_key = os.getenv("COHERE_API_KEY")
    use_cohere = bool(cohere_key) and not args.no_cohere
    section("Embedding provider", "🧠", color_on)
    if use_cohere:
        kv("Provider", "Cohere (v2)", color_on)
    else:
        kv(
            "Provider",
            f"Local fallback (dim={args.dim}){' — COHERE_API_KEY ignored' if cohere_key and args.no_cohere else ''}",
            color_on,
        )

    def embed_text(text: str) -> List[float]:
        if use_cohere:
            return client.embed_cohere(text).get("embedding", [])
        return hashed_bow_embedding(text, dim=args.dim)

    # 1) Create library
    section("Library CRUD", "📚", color_on)
    lib_name = f"demo-lib-{int(time.time())}"
    t = timeit("create_library")
    library = client.create_library(
        name=lib_name,
        description="Demo library created by e2e script",
        metadata={"owner": "e2e", "env": "demo"},
    )
    kv("Create", f"{elapsed(t)}", color_on, key_color=Palette.GREEN)
    kv("Library ID", library["id"], color_on)
    library_id = library["id"]

    # 1a) Update and get library
    t = timeit("update_library")
    client.update_library(library_id, description="Updated description")
    kv("Update", elapsed(t), color_on, key_color=Palette.GREEN)
    t = timeit("get_library")
    _ = client.get_library(library_id)
    kv("Get", elapsed(t), color_on, key_color=Palette.GREEN)

    # 2) Create a document
    section("Document CRUD", "📄", color_on)
    t = timeit("create_document")
    document = client.create_document(
        library_id=library_id,
        title="Demo Document",
        description="A sample document for e2e",
        metadata={"category": "demo"},
    )
    document_id = document["id"]
    kv("Create", f"{elapsed(t)}", color_on, key_color=Palette.GREEN)
    kv("Document ID", document_id, color_on)

    # 2a) Update document
    t = timeit("update_document")
    _ = client.update_document(library_id, document_id, title="Demo Document (updated)")
    kv("Update", elapsed(t), color_on, key_color=Palette.GREEN)

    # 3) Create chunks with embeddings
    section("Chunk creation", "✂️", color_on)
    texts: List[str] = [
        "The Eiffel Tower is located in Paris.",
        "Mount Everest is the tallest mountain on Earth.",
        "Python is a popular programming language for AI.",
        "Istanbul spans two continents: Europe and Asia.",
    ]
    kv("Count", str(len(texts)), color_on)
    created_chunk_ids: List[str] = []
    for idx, text in enumerate(texts, start=1):
        vec = embed_text(text)
        if not isinstance(vec, list) or not vec:
            raise RuntimeError("Embedding provider returned an empty vector")
        chunk = client.create_chunk(
            library_id=library_id,
            document_id=document_id,
            text=text,
            embedding=vec,
            metadata={"idx": str(idx), "lang": "en" if idx != 4 else "tr"},
        )
        created_chunk_ids.append(chunk["id"])
        kv("Chunk", f"created id={chunk['id']}", color_on, key_color=Palette.BLUE)

    # 3a) Update a chunk's metadata
    if created_chunk_ids:
        t = timeit("update_chunk")
        _ = client.update_chunk(
            library_id, created_chunk_ids[0], metadata={"tag": "updated"}
        )
        kv("Update first chunk", elapsed(t), color_on, key_color=Palette.GREEN)

    # 4) Search BEFORE building index (fallback linear)
    section("Search (no index: fallback linear)", "🔎", color_on)
    query = "Where is the Eiffel Tower?"
    qvec = embed_text(query)
    kv("Query", query, color_on)
    t = timeit("search_fallback_linear")
    res = client.search(library_id=library_id, vector=qvec, k=3)
    kv("Timing", elapsed(t), color_on, key_color=Palette.GREEN)
    for line in pretty_results(res.get("results", []), enabled=color_on):
        print(line)

    # 4a) Search with metadata filter
    section("Search with metadata filter (lang=en)", "🧹", color_on)
    kv("Filter", "lang=en", color_on)
    t = timeit("search_with_filter")
    res = client.search(
        library_id=library_id, vector=qvec, k=5, metadata_filters={"lang": "en"}
    )
    kv("Timing", elapsed(t), color_on, key_color=Palette.GREEN)
    for line in pretty_results(res.get("results", []), enabled=color_on):
        print(line)

    # 5) Build index (linear + cosine)
    section("Build index: linear (cosine)", "⚙️", color_on)
    t = timeit("build_index_linear_cosine")
    info = client.build_index(
        library_id=library_id, algorithm="linear", metric="cosine"
    )
    kv("Timing", elapsed(t), color_on, key_color=Palette.GREEN)
    kv("Algorithm", info.get("algorithm", ""), color_on)
    kv("Metric", info.get("metric", ""), color_on)
    t = timeit("search_linear_cosine")
    res = client.search(library_id=library_id, vector=qvec, k=3)
    kv("Search timing", elapsed(t), color_on, key_color=Palette.GREEN)
    for line in pretty_results(res.get("results", []), enabled=color_on):
        print(line)

    # 6) Build index (KD-Tree, euclidean)
    section("Build index: kdtree (euclidean)", "🌳", color_on)
    t = timeit("build_index_kdtree_euclidean")
    info = client.build_index(
        library_id=library_id, algorithm="kdtree", metric="euclidean"
    )
    kv("Timing", elapsed(t), color_on, key_color=Palette.GREEN)
    kv("Algorithm", info.get("algorithm", ""), color_on)
    kv("Metric", info.get("metric", ""), color_on)
    t = timeit("search_kdtree_euclidean")
    res = client.search(library_id=library_id, vector=qvec, k=3)
    kv("Search timing", elapsed(t), color_on, key_color=Palette.GREEN)
    for line in pretty_results(res.get("results", []), enabled=color_on):
        print(line)

    # 7) Build index (LSH, cosine)
    section("Build index: lsh (cosine)", "🧭", color_on)
    t = timeit("build_index_lsh_cosine")
    info = client.build_index(library_id=library_id, algorithm="lsh", metric="cosine")
    kv("Timing", elapsed(t), color_on, key_color=Palette.GREEN)
    kv("Algorithm", info.get("algorithm", ""), color_on)
    kv("Metric", info.get("metric", ""), color_on)
    t = timeit("search_lsh_cosine")
    res = client.search(library_id=library_id, vector=qvec, k=3)
    kv("Search timing", elapsed(t), color_on, key_color=Palette.GREEN)
    for line in pretty_results(res.get("results", []), enabled=color_on):
        print(line)

    # 8) Save and reload snapshot
    section("Persistence: save and load", "💾", color_on)
    t = timeit("admin_save")
    client.save()
    kv("Save", elapsed(t), color_on, key_color=Palette.GREEN)
    kv("Load", "starting...", color_on)
    t = timeit("admin_load")
    client.load()
    kv("Load", elapsed(t), color_on, key_color=Palette.GREEN)

    # 9) Optional cleanup to demonstrate delete endpoints
    if args.cleanup:
        section("Cleanup", "🧽", color_on)
        if created_chunk_ids:
            client.delete_chunk(library_id, created_chunk_ids[0])
            kv("Delete chunk", created_chunk_ids[0], color_on, key_color=Palette.YELLOW)
        client.delete_document(library_id, document_id)
        kv("Delete document", document_id, color_on, key_color=Palette.YELLOW)
        client.delete_library(library_id)
        kv("Delete library", library_id, color_on, key_color=Palette.YELLOW)

    section("Summary", "✅", color_on)
    print("  - Created library, document, chunks; updated library/document/chunk")
    print("  - Searched without index (fallback linear) and with metadata filter")
    print("  - Built and searched with linear, kdtree, and lsh indices")
    print("  - Saved and loaded snapshot")
    if args.cleanup:
        print("  - Cleaned up resources (delete endpoints)")
    print(c("Demo complete.", Palette.GREEN + Palette.BOLD, color_on))


if __name__ == "__main__":
    main()
