"""Local Chroma index, explicit ingestion, and read-only retrieval."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]


def normalize_query(query):
    """Bilingual terminology expansion, not an incident-type classifier.

    Korean 'memory' is otherwise frequently confused with storage in cross-lingual
    retrieval. No information/incident ID is chosen here; all types come from hits.
    """
    glossary=[(r"메모리|\bRAM\b", "RAM physical memory"),
              (r"로그", "application system logs"),
              (r"디스크|저장\s*공간|파일시스템", "disk filesystem storage space"),
              (r"\bcpu\b|프로세서|연산\s*부하", "CPU processor utilization")]
    words=[meaning for pattern,meaning in glossary if re.search(pattern,query,re.I)]
    return query + ("\nTerminology: " + "; ".join(words) if words else "")


class ManualUnavailable(ValueError):
    pass


class OllamaEmbedder:
    def __init__(self, model=None):
        self.model = model or os.environ.get("MANUAL_EMBED_MODEL", "bge-m3")
        self.url = os.environ.get("MANUAL_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        parsed = urlsplit(self.url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ManualUnavailable("임베딩 서버는 로컬 HTTP 주소만 지원합니다.")

    def embed(self, texts):
        try:
            response = httpx.post(self.url + "/api/embed", json={"model": self.model, "input": texts,
                                  "truncate": False, "keep_alive": "10m"}, timeout=120, trust_env=False)
            response.raise_for_status()
            vectors = response.json()["embeddings"]
            if len(vectors) != len(texts) or not all(vectors): raise ValueError("Empty embeddings")
            return vectors
        except (httpx.HTTPError, KeyError, ValueError):
            raise ManualUnavailable("로컬 매뉴얼 임베딩을 생성하지 못했습니다. Ollama와 bge-m3 모델을 확인해주세요.") from None


class ManualIndex:
    def __init__(self, root=ROOT, embedder=None, index_dir=None):
        self.root = Path(root)
        self.base = self.root / "data/manuals"
        self.path = Path(index_dir or os.environ.get("MANUAL_INDEX_DIR", self.root / "data/manual_index"))
        self.embedder = embedder or OllamaEmbedder()
        self._client = None

    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
            except Exception as exc:
                raise ManualUnavailable("로컬 매뉴얼 벡터 저장소를 열지 못했습니다.") from exc
        return self._client

    def _prune_obsolete(self, active_name):
        # Only this module's derived collections, never unrelated user collections.
        client=self.client()
        for old in client.list_collections():
            old_name=old if isinstance(old,str) else old.name
            if old_name.startswith("official-") and old_name != active_name:
                client.delete_collection(old_name)

    def build(self):
        chunks = json.loads((self.base / "chunks.json").read_text())
        from .actions import candidates
        corpus_hash = hashlib.sha256((self.base / "chunks.json").read_bytes()).hexdigest()
        stamp = hashlib.sha256(json.dumps({"chunks":chunks, "model":self.embedder.model, "index_schema":4,
                                          "reviewed_chunks":[c['chunk_id'] for c in chunks if candidates([c])]}, sort_keys=True).encode()).hexdigest()
        marker = self.path / "active.json"
        previous = json.loads(marker.read_text()) if marker.exists() else None
        if previous and previous["fingerprint"] == stamp:
            self.client().get_collection(previous["collection"])
            self._prune_obsolete(previous["collection"])
            return {"changed": False, "chunks": len(chunks)}
        name = "official-" + stamp[:24]
        client = self.client()
        collection = client.get_or_create_collection(name, metadata={"hnsw:space":"cosine"}, embedding_function=None)
        existing = {}
        if previous and previous.get("model") == self.embedder.model:
            old = client.get_collection(previous["collection"]).get(include=["embeddings", "metadatas"])
            for cid, meta, vector in zip(old["ids"], old["metadatas"], old["embeddings"]):
                existing[cid] = (meta["text_hash"], vector.tolist())
        new_embeddings, reused = 0, 0
        for start in range(0, len(chunks), 8):
            batch = chunks[start:start+8]
            text = ["Incident metadata: " + ", ".join(c.get("incident_types",[])).replace("-"," ") + "\n" + c["title"] + "\n" + c["section"] + "\n" + c["text"] for c in batch]
            hashes = [hashlib.sha256(t.encode()).hexdigest() for t in text]
            missing = [i for i,c in enumerate(batch) if c["chunk_id"] not in existing or existing[c["chunk_id"]][0] != hashes[i]]
            generated = iter(self.embedder.embed([text[i] for i in missing])) if missing else iter([])
            vectors = [next(generated) if i in missing else existing[c["chunk_id"]][1] for i,c in enumerate(batch)]
            new_embeddings += len(missing); reused += len(batch) - len(missing)
            metadata = [{"os":c["os"], "portable":c["portable"], "product":c["product"], "version":c["version"],
                         "doc_id":c["doc_id"], "has_actions":bool(candidates([c])), "text_hash":hashes[i], "chunk_json":json.dumps(c,ensure_ascii=False)} for i,c in enumerate(batch)]
            collection.upsert(ids=[c["chunk_id"] for c in batch], embeddings=vectors, documents=text, metadatas=metadata)
            print("indexed", min(start+8,len(chunks)), "/", len(chunks), flush=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        temp = marker.with_suffix(".tmp")
        temp.write_text(json.dumps({"fingerprint":stamp, "collection":name, "model":self.embedder.model, "chunks":len(chunks),"corpus_sha256":corpus_hash}, indent=2))
        temp.replace(marker)
        # Only after a successful replacement, remove obsolete derived official
        # collections so deleted manuals do not remain in the persistent index.
        self._prune_obsolete(name)
        return {"changed":True, "chunks":len(chunks), "embedded":new_embeddings, "reused":reused}

    def search(self, query, os_name, evidence_summary="", product=None, minimum_score=None, allow_mixed_topics=False):
        if os_name not in {"linux", "macos"}: raise ValueError("지원 OS는 linux 또는 macos입니다.")
        if not query.strip() or len(query)>4000 or len(evidence_summary)>4000: raise ValueError("검색 질의 길이를 확인해주세요.")
        begin = perf_counter()
        marker = self.path / "active.json"
        if not marker.exists(): raise ManualUnavailable("매뉴얼 인덱스가 없습니다. 수집·색인 명령을 먼저 실행해주세요.")
        active = json.loads(marker.read_text())
        if active["model"] != self.embedder.model: raise ManualUnavailable("임베딩 모델이 바뀌었습니다. 매뉴얼을 다시 색인해주세요.")
        if active.get("corpus_sha256") != hashlib.sha256((self.base / "chunks.json").read_bytes()).hexdigest():
            raise ManualUnavailable("매뉴얼 본문이 바뀌었습니다. 다시 색인하고 검색 기준을 보정해주세요.")
        settings = json.loads((self.base / "retrieval.json").read_text())
        if minimum_score is None:
            if settings.get("status","calibrated")!="calibrated" or settings.get("embedding_model",self.embedder.model)!=self.embedder.model or settings.get("index_fingerprint",active["fingerprint"])!=active["fingerprint"]:
                raise ManualUnavailable("현재 매뉴얼 인덱스의 검색 기준을 다시 보정해주세요.")
        if minimum_score is None: minimum_score = settings.get("minimum_score_by_os",{}).get(os_name,settings["minimum_score"])
        # Mixed numeric observations can overwhelm the requested symptom (e.g.
        # top includes RAM figures even when the user asks only about CPU).
        # Keep retrieval aligned with the current question. Observations remain
        # available to the downstream applicability/step selection model.
        vector = self.embedder.embed([normalize_query(query)])[0]
        embedded_at = perf_counter()
        filters = {"$and":[{"os":os_name}, {"portable":True}]} if not product else {"$and":[{"os":os_name},{"$or":[{"portable":True},{"product":product}]}]}
        if settings.get("require_reviewed_actions", True): filters["$and"].append({"has_actions":True})
        try:
            collection = self.client().get_collection(active["collection"], embedding_function=None)
            result = collection.query(query_embeddings=[vector], where=filters, n_results=min(5,active["chunks"]), include=["metadatas", "distances"])
        except Exception as exc:
            raise ManualUnavailable("매뉴얼 검색에 실패했습니다. 기존 서버 조회 결과는 유지됩니다.") from exc
        from .actions import candidates
        ranked=[]
        for meta, distance in zip(result["metadatas"][0],result["distances"][0]):
            c = json.loads(meta["chunk_json"])
            score = round(1-distance, 6)
            if score>=minimum_score: ranked.append({**c,"score":score})
        # A nearest hit is not a reliable topic classifier. Keep the two best
        # distinct remedies; the selector checks applicability against the task.
        compatible=ranked
        first,duplicates,documents=[],[],set()
        for c in compatible:
            if c["doc_id"] in documents: duplicates.append(c)
            else: first.append(c);documents.add(c["doc_id"])
        matches, seen, seen_actions = [], set(), set()
        for c in first+duplicates:
            identity = hashlib.sha256(c["text"].encode()).hexdigest()
            if identity in seen: continue
            action_ids={a['action_id'] for a in candidates([c])}
            if action_ids and action_ids <= seen_actions: continue
            seen.add(identity)
            seen_actions.update(action_ids)
            matches.append({**c, "conditions":"문서의 적용 환경·전제조건을 확인해야 하며 실제 조치는 실행하지 않습니다."})
            if len(matches)==2: break
        return {"matches":matches, "model":self.embedder.model, "minimum_score":minimum_score,
                "timings_ms":{"embedding":round((embedded_at-begin)*1000,2), "search":round((perf_counter()-embedded_at)*1000,2)}}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--build", action="store_true")
    args=parser.parse_args()
    if args.build: print(json.dumps(ManualIndex().build(),ensure_ascii=False))
