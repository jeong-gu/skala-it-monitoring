"""Separate development calibration from held-out retrieval/answer evaluation."""
import argparse
import asyncio
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from .index import ManualIndex, ROOT


def cases():
    return json.loads((ROOT/"eval/remediation_cases.json").read_text())["cases"]


def calibrate(index):
    rows=[]
    for c in cases():
        if c["split"]!="development" or not c.get("os"): continue
        result=index.search(c["query"],c["os"],minimum_score=0)
        rows.append({"id":c["id"],"query":c["query"],"os":c["os"],"positive":c["kind"]=="positive","expected":c.get("expected_docs",[]),
                     "matches":[{"doc_id":m["doc_id"],"score":m["score"]} for m in result["matches"]]})
    thresholds,hit={},0
    for os_name in ["linux","macos"]:
        subset=[r for r in rows if r["os"]==os_name]
        scores={m["score"] for row in subset for m in row["matches"]}
        choices=sorted({0.0,1.0,*(round(s+0.000001,6) for s in scores)})
        valid=[]
        for threshold in choices:
            if any(any(m["score"]>=threshold for m in row["matches"]) for row in subset if not row["positive"]): continue
            hits=sum(any(m["score"]>=threshold and m["doc_id"] in row["expected"] for m in row["matches"]) for row in subset if row["positive"])
            valid.append((hits,-threshold))
        hits,negative_threshold=max(valid);hit+=hits;thresholds[os_name]=-negative_threshold
    setting={"schema_version":1,"minimum_score":max(thresholds.values()),"minimum_score_by_os":thresholds,"status":"calibrated","require_reviewed_actions":True,
             "index_fingerprint":json.loads((index.path/"active.json").read_text())["fingerprint"],
             "embedding_model":index.embedder.model,"candidate_count":5,"result_chunk_limit":2,
             "development_hits":hit,"development_positive_count":sum(r["positive"] for r in rows),
             "development_false_acceptances":0,"calibrated_at":datetime.now(timezone.utc).isoformat()}
    return setting,rows


class EvaluationMCP:
    def __init__(self,index): self.index=index; self.calls=[]
    async def call_tool(self,name,args):
        self.calls.append(name)
        if name=="search_remediation_manuals": return await asyncio.to_thread(self.index.search,args["query"],args["os"],args.get("evidence_summary",""))
        if name=="list_command_servers": return [{"server_id":"server-g","os":"macos"}]
        raise AssertionError("Generic manual question must not query server commands: "+name)


async def evaluate(index, model=None):
    from agent.remediation import remediation_node,wants_remediation
    rows=[]
    for c in cases():
        begin=perf_counter(); row={"id":c["id"],"query":c["query"],"split":c["split"],"kind":c["kind"]}
        if c["kind"]=="no_trigger":
            row["passed"]=not wants_remediation(c["query"])
        elif model is None:
            if not c.get("os"): continue
            result=index.search(c["query"],c["os"])
            row["matches"]=[{"doc_id":m["doc_id"],"score":m["score"]} for m in result["matches"]]
            row["passed"]=any(m["doc_id"] in c.get("expected_docs",[]) for m in result["matches"]) if c["kind"]=="positive" else not result["matches"]
        else:
            mcp=EvaluationMCP(index)
            result=await remediation_node({"query":c["query"],"history":c.get("history",[]),"events":[]},SimpleNamespace(mcp=mcp,quality_llm=model))
            answer=result["remediation"]
            row.update(status=answer["status"],steps=answer["steps"],sources=answer["sources"],calls=mcp.calls,
                       timings_ms=answer.get("timings_ms",{}),reply=result["final"])
            if c["kind"]=="positive":
                row["passed"]=answer["status"]=="provided" and any(s["doc_id"] in c["expected_docs"] for s in answer["sources"]) and all(s["os"]==c["os"] for s in answer["sources"]) and all(c["expected_incident"] in s["incident_types"] for s in answer["steps"])
            else:
                row["passed"]=answer["status"]==c["expected_status"]
            if c.get("forbid_manual_search"):
                row["passed"] &= "search_remediation_manuals" not in mcp.calls
        row["seconds"]=round(perf_counter()-begin,3);rows.append(row)
        print(c["id"],row["passed"],row["seconds"],flush=True)
    return {"created_at":datetime.now(timezone.utc).isoformat(),"backend":"ollama" if model else "retrieval_only",
            "model":getattr(model,"name",None),"suite_sha256":hashlib.sha256((ROOT/"eval/remediation_cases.json").read_bytes()).hexdigest(),
            "index":json.loads((index.path/"active.json").read_text()),
            "passed":sum(r["passed"] for r in rows),"total":len(rows),"cases":rows}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--calibrate",action="store_true");p.add_argument("--ollama",action="store_true");p.add_argument("--output",required=True)
    args=p.parse_args();index=ManualIndex()
    if args.calibrate:
        settings,rows=calibrate(index)
        (ROOT/"data/manuals/retrieval.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2))
        report={"settings":settings,"development_only":rows}
    else:
        model=None
        if args.ollama:
            from agent.ollama_llm import OllamaQualityLLM
            model=OllamaQualityLLM("qwen3:8b",0)
        report=asyncio.run(evaluate(index,model))
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
