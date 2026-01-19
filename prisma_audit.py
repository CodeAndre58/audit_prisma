from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from models import *

console = Console()

# ----------------------------
# Utilities
# ----------------------------
def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    cleaned = soup.get_text(separator=" ", strip=True)
    return normalize_ws(cleaned)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    v = d.get(key, default)
    return default if v is None else v

def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_papers_from_json(payload: Any) -> List[Dict[str, Any]]:
    """
    Supporta:
    - lista di dict (il tuo caso)
    - dict con chiave tipo "results"/"items"/"data"
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("results", "items", "data", "papers"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    raise ValueError("Formato JSON non riconosciuto: mi aspettavo una lista di paper o un dict con lista in una chiave nota.")

def parse_paper(raw: Dict[str, Any]) -> Paper:
    pid = str(safe_get(raw, "Id", "")) or str(safe_get(raw, "id", ""))
    doi = str(safe_get(raw, "Doi", "")) or str(safe_get(raw, "doi", ""))
    title = normalize_ws(str(safe_get(raw, "Title", "")) or str(safe_get(raw, "title", "")))
    abstract_raw = str(safe_get(raw, "Abstract", "")) or str(safe_get(raw, "abstract", ""))
    abstract = clean_html(abstract_raw)
    year = safe_get(raw, "PublicationYear", None)
    portal = safe_get(raw, "Portal", None)
    authors = safe_get(raw, "Authors", None)
    if isinstance(authors, str):
        authors = [authors]
    if not isinstance(authors, list):
        authors = None
    return Paper(id=pid, doi=doi, title=title, abstract=abstract, year=year, portal=portal, authors=authors)

def paper_text(p: Paper) -> str:
    # Nella fase “limitata” lavoriamo SOLO su titolo+abstract
    return normalize_ws(f"TITLE: {p.title}\nABSTRACT: {p.abstract}")

def clip_quotes(snips: List[str], max_len: int = 420) -> List[str]:
    out = []
    for s in snips:
        s = normalize_ws(s)
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        out.append(s)
    return out[:3]

# ----------------------------
# Heuristic checker
# ----------------------------
def heuristic_check(item: PrismaItem, text: str) -> ItemResult:
    t = text.lower()

    hits = [kw for kw in item.keywords if kw.lower() in t]
    evidence = []

    # prova a estrarre 1-2 frasi attorno alla keyword
    if hits:
        for kw in hits[:2]:
            m = re.search(re.escape(kw), t)
            if not m:
                continue
            start = max(0, m.start() - 120)
            end = min(len(t), m.end() + 200)
            snippet = text[start:end]  # usa testo originale per non perdere maiuscole
            evidence.append(snippet)

    # verdict base
    if item.item_id == "1":
        # più rigido: in title
        in_title = any(kw in (text.split("\n", 1)[0].lower()) for kw in item.keywords)
        verdict: Verdict = "present" if in_title else ("unclear" if hits else "absent")
        conf = 0.80 if in_title else (0.55 if hits else 0.65)
        notes = "Heuristic: controllo keyword nel titolo (forte) e nel testo complessivo (debole)."
        return ItemResult(item.item_id, item.name, verdict, conf, clip_quotes(evidence), notes, method="heuristic")

    # per gli altri item: hits nel titolo+abstract
    if hits:
        verdict = "present"
        conf = 0.65
        notes = f"Heuristic: trovate keyword {hits[:4]} nel testo (titolo+abstract)."
    else:
        verdict = "absent"
        conf = 0.55
        notes = "Heuristic: nessuna keyword trovata (ma l'abstract potrebbe non contenere la sezione richiesta)."

    # Se item tipicamente non sta nell'abstract, preferisci 'unclear' quando è plausibile
    if item.item_id in {"25", "26", "27", "24"} and not hits:
        verdict = "unclear"
        conf = 0.50
        notes = "Heuristic: non trovato nell'abstract; spesso appare solo nel full-text (quindi 'unclear')."

    return ItemResult(item.item_id, item.name, verdict, conf, clip_quotes(evidence), notes, method="heuristic")

# ----------------------------
# LLM judgement (optional)
# ----------------------------
class LLMJudgement(BaseModel):
    verdict: Verdict = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_quotes: List[str] = Field(default_factory=list)
    rationale: str = Field(...)

def extract_json_object(s: str) -> str:
    """
    Estrae un oggetto JSON dalla stringa (fallback robusto).
    """
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # cerca prima { ... } più esterno
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s

class LLMClient:
    def judge(self, item: PrismaItem, paper: Paper, heuristic: ItemResult) -> ItemResult:
        raise NotImplementedError

class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def judge(self, item: PrismaItem, paper: Paper, heuristic: ItemResult) -> ItemResult:
        prompt = build_judge_prompt(item, paper, heuristic)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict PRISMA compliance auditor. Use only the provided text. Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }

        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "") or ""
        return parse_llm_output(item, heuristic, content)

def build_judge_prompt(item: PrismaItem, paper: Paper, heuristic: ItemResult) -> str:
    txt = paper_text(paper)
    return (
        f"ITEM:\n- id: {item.item_id}\n- name: {item.name}\n- definition: {item.definition}\n\n"
        f"PAPER TEXT (title + abstract only):\n{txt}\n\n"
        f"HEURISTIC PRE-CHECK:\n"
        f"- verdict: {heuristic.verdict}\n- confidence: {heuristic.confidence}\n- evidence: {heuristic.evidence}\n- notes: {heuristic.notes}\n\n"
        "TASK:\n"
        "Decide whether the ITEM is satisfied using ONLY the PAPER TEXT above.\n"
        "If the evidence is insufficient (common with abstract-only), return verdict='unclear'.\n\n"
        "Return ONLY this JSON (no markdown):\n"
        "{\n"
        "  \"verdict\": \"present|absent|unclear\",\n"
        "  \"confidence\": 0.0,\n"
        "  \"evidence_quotes\": [\"short quote 1\", \"short quote 2\"],\n"
        "  \"rationale\": \"1-3 sentences\"\n"
        "}\n"
    )

def parse_llm_output(item: PrismaItem, heuristic: ItemResult, content: str) -> ItemResult:
    raw = extract_json_object(content)
    try:
        obj = json.loads(raw)
        parsed = LLMJudgement(**obj)
        evidence = clip_quotes(parsed.evidence_quotes)
        notes = f"LLM rationale: {parsed.rationale}"
        return ItemResult(item.item_id, item.name, parsed.verdict, float(parsed.confidence), evidence, notes, method="llm")
    except (json.JSONDecodeError, ValidationError) as e:
        # fallback: se il parsing fallisce, tieni heuristic
        return ItemResult(
            item.item_id,
            item.name,
            heuristic.verdict,
            max(0.30, heuristic.confidence - 0.10),
            heuristic.evidence,
            f"LLM output non parsabile, fallback heuristic. Errore: {type(e).__name__}",
            method="heuristic",
        )

# ----------------------------
# Runner
# ----------------------------
def collect_papers(input_patterns: List[str]) -> List[Paper]:
    files: List[str] = []
    for pat in input_patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError("Nessun file trovato. Controlla --input (pattern glob).")

    papers: List[Paper] = []
    for fp in files:
        payload = load_json_any(fp)
        raw_list = iter_papers_from_json(payload)
        for raw in raw_list:
            p = parse_paper(raw)
            # filtra record vuoti
            if p.title or p.abstract:
                papers.append(p)
    return papers

def audit_paper(p: Paper, llm: Optional[LLMClient]) -> PaperReport:
    results: List[ItemResult] = []
    txt = paper_text(p)

    for item in ITEMS:
        h = heuristic_check(item, txt)
        if llm is None:
            results.append(h)
        else:
            results.append(llm.judge(item, p, h))

    return PaperReport(paper_id=p.id, title=p.title, doi=p.doi, results=results)

def save_outputs(reports: List[PaperReport], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSONL (una riga per paper)
    jsonl_path = out / "report.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rep in reports:
            f.write(json.dumps({
                "paper_id": rep.paper_id,
                "title": rep.title,
                "doi": rep.doi,
                "results": [asdict(r) for r in rep.results],
            }, ensure_ascii=False) + "\n")

    # CSV (una riga per item per paper)
    rows = []
    for rep in reports:
        for r in rep.results:
            rows.append({
                "paper_id": rep.paper_id,
                "doi": rep.doi,
                "title": rep.title,
                "item_id": r.item_id,
                "item_name": r.item_name,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "method": r.method,
                "evidence": " | ".join(r.evidence),
                "notes": r.notes,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "report.csv", index=False, encoding="utf-8")

def build_llm(args) -> Optional[LLMClient]:
    mode = (args.llm or "").lower() if isinstance(args.llm, str) else args.llm
    if mode == "none":
        return None
    if mode == "ollama":
        return OllamaClient(base_url=args.ollama_url, model=args.model, timeout=args.timeout)

    raise ValueError(f"--llm non valido: {args.llm}")

def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="PRISMA abstract-level audit (title + abstract from JSON).")
    ap.add_argument("--input", nargs="+", required=True, help="Uno o più pattern glob (es: input/*.json)")
    ap.add_argument("--out", default="output", help="Cartella output")
    ap.add_argument("--llm", choices=["none", "ollama", "openai"], default="none", help="LLM backend ('none' per disabilitare)")
    ap.add_argument("--model", default="llama3.1", help="Nome modello (ollama) o modello OpenAI")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL Ollama")
    ap.add_argument("--timeout", type=int, default=60, help="Timeout chiamate LLM (secondi)")
    args = ap.parse_args()

    llm = build_llm(args)

    papers = collect_papers(args.input)
    console.print(f"[bold]Trovati {len(papers)} paper[/bold] dai JSON.")

    reports: List[PaperReport] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analisi in corso...", total=len(papers))
        for p in papers:
            rep = audit_paper(p, llm)
            reports.append(rep)
            progress.advance(task)

    save_outputs(reports, args.out)

    # Mini riassunto
    df = pd.read_csv(Path(args.out) / "report.csv")
    summary = df.groupby(["item_id", "item_name", "verdict"]).size().reset_index(name="count")
    console.print("\n[bold]Summary (counts per item/verdict):[/bold]")
    console.print(summary.to_string(index=False))
    console.print(f"\nOutput salvati in: [bold]{args.out}[/bold] (report.jsonl, report.csv)")

if __name__ == "__main__":
    main()
