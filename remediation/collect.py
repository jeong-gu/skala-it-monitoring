"""Explicit download/parse/chunk CLI. Never imported by the chat execution path."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import httpx
from bs4 import BeautifulSoup
from tokenizers import Tokenizer

from .sources import SOURCES

ROOT = Path(__file__).resolve().parents[1]


def sections(raw, spec):
    if spec.get("format") == "roff":
        result = []
        for block in re.split(r"(?m)^\.SH ", raw):
            lines = block.splitlines()
            if not lines: continue
            title = lines[0].strip('" ')
            text = "\n".join(lines[1:])
            text = re.sub(r"\\f[BRIP]|\\&|\\%", "", text).replace(r"\-", "-")
            text = re.sub(r"(?m)^\.(?:B|I|BR|IR|BI|RB|RI)\s+", "", text)
            text = re.sub(r"(?m)^\.[A-Za-z].*$", "", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text and title in {"DESCRIPTION", "CONFIGURATION FILE"}: result.append((title, text))
        return result
    soup = BeautifulSoup(raw, "html.parser")
    if spec.get("definition_terms"):
        # Keep each configuration procedure together with its official cautions.
        result=[]
        for term in spec["definition_terms"]:
            node=next((n for n in soup.select("dt") if n.get_text(" ",strip=True)==term),None)
            detail=node.find_next_sibling("dd") if node is not None else None
            if detail is None: raise ValueError(f"Missing configuration section: {term}")
            result.append((term," ".join(detail.get_text(" ",strip=True).split())))
        return result
    main = soup.select_one("main") or soup.select_one("#content") or soup.select_one("article") or soup.select_one("[role=main]") or soup.body
    if main is None: raise ValueError(f"No document body: {spec['doc_id']}")
    # Red Hat keeps the book's navigation in main; only read actual chapter content.
    body = main.select_one("article") or main.select_one("section.chapter") or main
    for el in body.select("script,style,nav,footer,header,.visuallyhidden,.a11y"):
        el.decompose()
    result, heading, paragraphs = [], spec["title"], []
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "dt", "dd"]):
        if el.find_parent(["li", "pre", "dd"]): continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text: continue
        if el.name.startswith("h"):
            if paragraphs: result.append((heading, "\n\n".join(paragraphs)))
            heading, paragraphs = re.sub(r"\s+Copy link.*$", "", text), []
        else:
            paragraphs.append(text)
    if paragraphs: result.append((heading, "\n\n".join(paragraphs)))
    result = [(h,t) for h,t in result if h not in {"Activity Monitor User Guide", "Mac User Guide"} and not t.startswith("Next:")]
    if spec.get("section_filter"):
        result = [(h, t) for h, t in result if spec["section_filter"].lower() in h.lower()]
    if not result: raise ValueError(f"Empty body: {spec['doc_id']}")
    return result


def chunk_section(text, tokenizer, maximum=600, overlap=80):
    """Keep complete paragraphs/procedures where possible; token offsets preserve text."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) <= maximum:
        return [(text, len(encoded.ids))]
    result, start = [], 0
    while start < len(encoded.ids):
        end = min(start + maximum, len(encoded.ids))
        # Prefer paragraph boundary in the latter half, without breaking the token cap.
        left, right = encoded.offsets[start][0], encoded.offsets[end - 1][1]
        boundary = text.rfind("\n\n", encoded.offsets[min(start + maximum // 2, end - 1)][0], right)
        if end < len(encoded.ids) and boundary > left:
            while end > start and encoded.offsets[end - 1][1] > boundary: end -= 1
            right = encoded.offsets[end - 1][1]
        part = text[left:right].strip()
        if part: result.append((part, len(tokenizer.encode(part, add_special_tokens=False).ids)))
        if end == len(encoded.ids): break
        start = max(start + 1, end - overlap)
    return result


def collect(root=ROOT, refresh=False):
    from .actions import candidates
    base = root / "data/manuals"
    old_manifest=base / "manifest.json"
    previous=json.loads(old_manifest.read_text())["documents"] if old_manifest.exists() else []
    for directory in [base / "originals", base / "text", root / ".manual_cache"]:
        directory.mkdir(parents=True, exist_ok=True)
    token_path = root / ".manual_cache/bge-m3-tokenizer.json"
    with httpx.Client(timeout=90, follow_redirects=True, trust_env=False) as client:
        if not token_path.exists():
            response = client.get("https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json")
            response.raise_for_status(); token_path.write_bytes(response.content)
        tokenizer = Tokenizer.from_file(str(token_path))
        documents, chunks = [], []
        for spec in sorted(SOURCES, key=lambda item: item['doc_id'].startswith('rhel')):
            path = base / "originals" / (spec["doc_id"] + (".roff" if spec.get("format") == "roff" else ".html"))
            if refresh or not path.exists():
                download_url = spec["url"] + ("?download=1" if spec['doc_id'].startswith('rhel') else "")
                response = client.get(download_url); response.raise_for_status()
                path.write_bytes(response.content)
            raw = path.read_text()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            document = {**spec, "source_url": spec["url"], "collected_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        "sha256": digest, "original_path": str(path.relative_to(root))}
            documents.append(document)
            parsed = sections(raw, spec)
            text_path = base / "text" / (spec["doc_id"] + ".txt")
            text_path.write_text("\n\n".join(f"# {h}\n\n{t}" for h,t in parsed))
            for section_index, (heading, text) in enumerate(parsed):
                for part_index, (part, count) in enumerate(chunk_section(text, tokenizer)):
                    chunks.append({**{k:document[k] for k in ("doc_id", "os", "product", "version", "portable", "incident_types", "title", "source_url", "collected_at", "sha256")},
                                   "chunk_id": f"{spec['doc_id']}:{section_index}:{part_index}",
                                   "section": heading, "text": part, "token_count": count})
            print(spec["doc_id"], len(raw), "bytes", len(parsed), "sections", flush=True)
    # Index only sections carrying an actual reviewed remedy. Keep full originals
    # separately for provenance; lookup paragraphs never become search candidates.
    chunks=[c for c in chunks if candidates([c])]
    if not chunks: raise ValueError("No reviewed remediation chunks")
    (base / "manifest.json").write_text(json.dumps({"schema_version": 1, "tokenizer": "BAAI/bge-m3", "tokenizer_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(), "documents": documents}, ensure_ascii=False, indent=2))
    (base / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    # Remove only retired files recorded in the previous official manifest.
    retired={d["doc_id"] for d in previous}-{d["doc_id"] for d in documents}
    for doc in previous:
        if doc["doc_id"] not in retired: continue
        for path in [root/doc["original_path"],base/"text"/(doc["doc_id"]+".txt")]:
            if path.resolve().is_relative_to(base.resolve()): path.unlink(missing_ok=True)
    print(len(documents), "documents;", len(chunks), "chunks; max tokens", max(c["token_count"] for c in chunks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    collect(refresh=parser.parse_args().refresh)
