"""
prisma_audit.py

Script da riga di comando per eseguire un "PRISMA abstract-level audit"
su una collezione di articoli scientifici rappresentati in formato JSON
(titolo + abstract).

Usa:
- prisma_models.py per le strutture dati e helper base,
- prisma_verifier.py per la logica di verifica (euristica + LLM).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from models import Paper, PaperReport, clean_html, normalize_ws
from prisma_verifier import (
    LLMClient,
    OllamaClient,
    OpenAIChatClient,
    audit_paper,
    build_expanded_keywords,
)

console = Console()


# =============================================================================
# FUNZIONI DI CARICAMENTO/CONVERSIONE JSON → Paper
# =============================================================================


def safe_get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    """
    Recupera un valore da un dict gestendo chiavi mancanti o valori None.

    Args:
        d: Dizionario sorgente.
        key: Chiave da cercare.
        default: Valore di default se la chiave non esiste o è None.

    Returns:
        Valore del dizionario o default.
    """
    v = d.get(key, default)
    return default if v is None else v


def load_json_any(path: str) -> Any:
    """
    Carica un file JSON generico.

    Args:
        path: Percorso del file JSON.

    Returns:
        Oggetto Python corrispondente al contenuto JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_papers_from_json(payload: Any) -> List[Dict[str, Any]]:
    """
    Estrae una lista di "raw papers" dal contenuto JSON.

    Supporta due formati principali:
    - Lista di dict, ciascuno rappresenta un paper.
    - Dict che contiene una lista sotto chiavi comuni ("results", "items", "data", "papers").

    Args:
        payload: Oggetto Python risultante dal parsing JSON.

    Returns:
        Lista di dict, ciascuno un record di paper.

    Raises:
        ValueError: Se il formato non è riconosciuto.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("results", "items", "data", "papers"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    raise ValueError(
        "Formato JSON non riconosciuto: mi aspettavo una lista di paper "
        "o un dict con lista in una chiave nota."
    )


def parse_paper(raw: Dict[str, Any]) -> Paper:
    """
    Converte un dict grezzo (estratto dal JSON) in un oggetto Paper.

    Gestisce differenze di naming comuni (Id/id, Title/title, ecc.).

    Args:
        raw: Dizionario con i campi del paper.

    Returns:
        Istanza di Paper con campi normalizzati.
    """
    pid = str(safe_get(raw, "Id", "")) or str(safe_get(raw, "id", ""))
    doi = str(safe_get(raw, "Doi", "")) or str(safe_get(raw, "doi", ""))
    title = normalize_ws(
        str(safe_get(raw, "Title", "")) or str(safe_get(raw, "title", ""))
    )
    abstract_raw = str(safe_get(raw, "Abstract", "")) or str(
        safe_get(raw, "abstract", "")
    )
    abstract = clean_html(abstract_raw)
    year = safe_get(raw, "PublicationYear", None)
    portal = safe_get(raw, "Portal", None)
    authors = safe_get(raw, "Authors", None)
    if isinstance(authors, str):
        authors = [authors]
    if not isinstance(authors, list):
        authors = None
    return Paper(
        id=pid,
        doi=doi,
        title=title,
        abstract=abstract,
        year=year,
        portal=portal,
        authors=authors,
    )


def collect_papers(input_patterns: List[str]) -> List[Paper]:
    """
    Carica tutti i file JSON corrispondenti ai pattern indicati e li converte
    in una lista di Paper.

    Args:
        input_patterns: Lista di pattern glob (es. ["input/*.json"]).

    Returns:
        Lista di Paper con titolo/abstract puliti.

    Raises:
        FileNotFoundError: Se nessun file corrisponde ai pattern.
        ValueError: Se il formato dei JSON non è riconosciuto.
    """
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
            if p.title or p.abstract:
                papers.append(p)
    return papers


# =============================================================================
# OUTPUT RISULTATI
# =============================================================================


def save_outputs(reports: List[PaperReport], out_dir: str) -> None:
    """
    Salva i risultati dell'audit in formato JSONL e CSV nella cartella indicata.

    Args:
        reports: Lista di PaperReport da serializzare.
        out_dir: Percorso della cartella di output (verrà creata se non esiste).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "report.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rep in reports:
            f.write(
                json.dumps(
                    {
                        "paper_id": rep.paper_id,
                        "title": rep.title,
                        "doi": rep.doi,
                        "results": [asdict(r) for r in rep.results],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    rows: List[Dict[str, Any]] = []
    for rep in reports:
        for r in rep.results:
            rows.append(
                {
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
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "report.csv", index=False, encoding="utf-8")


# =============================================================================
# COSTRUZIONE CLIENT LLM
# =============================================================================


def build_llm(args: argparse.Namespace) -> Optional[LLMClient]:
    """
    Costruisce un client LLM in base ai parametri CLI.

    Args:
        args: Oggetto Namespace di argparse con gli argomenti già parsati.

    Returns:
        Istanza di LLMClient (OllamaClient, OpenAIChatClient) o None se --llm none.

    Raises:
        RuntimeError: Se mancano variabili d'ambiente necessarie (per OpenAI).
        ValueError: Se il valore di --llm non è riconosciuto.
    """
    if args.llm == "none":
        return None

    if args.llm == "ollama":
        return OllamaClient(
            base_url=args.ollama_url, model=args.model, timeout=args.timeout
        )

    if args.llm == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY non trovato. Mettilo in env o in .env per usare --llm openai."
            )
        return OpenAIChatClient(model=args.model)

    raise ValueError(f"--llm non valido: {args.llm}")


# =============================================================================
# MAIN / CLI
# =============================================================================


def main() -> None:
    """
    Funzione principale: gestisce input da riga di comando, esegue l'audit
    su tutti i paper e salva i risultati.
    """
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="PRISMA abstract-level audit (title + abstract from JSON)."
    )
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help='Uno o più pattern glob per i JSON (es: "input/*.json")',
    )
    ap.add_argument(
        "--out", default="output", help="Cartella in cui salvare i risultati"
    )
    ap.add_argument(
        "--llm",
        choices=["none", "ollama", "openai"],
        default="none",
        help=(
            "Backend LLM per il giudizio sugli item "
            "(none = solo euristiche, ollama = server locale, openai = API OpenAI)"
        ),
    )
    ap.add_argument(
        "--model",
        default="llama3.1",
        help="Nome modello (Ollama) o modello OpenAI (se --llm openai)",
    )
    ap.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL Ollama (usato per normalizzazione/espansione keyword e LLM=ollama)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in secondi per le chiamate LLM (giudizio, normalizzazione, espansione keyword)",
    )
    ap.add_argument(
        "--expand-keywords",
        action="store_true",
        help=(
            "Usa l'LLM (Ollama) per espandere le keyword degli item PRISMA "
            "(risultato memorizzato in expanded_keywords.json)."
        ),
    )
    ap.add_argument(
        "--normalize-with-llm",
        action="store_true",
        help=(
            "Usa l'LLM (Ollama) per normalizzare il testo di ciascun paper "
            "prima del matching euristico."
        ),
    )
    args = ap.parse_args()

    llm = build_llm(args)

    expanded_kw_map: Optional[Dict[str, List[str]]] = None
    if args.expand_keywords:
        expanded_kw_map = build_expanded_keywords(
            base_url=args.ollama_url,
            model=args.model,
            timeout=args.timeout,
            cache_path="expanded_keywords.json",
        )

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
            rep = audit_paper(
                p,
                llm,
                expanded_kw_map=expanded_kw_map,
                normalize_with_llm=args.normalize_with_llm,
                ollama_base_url=args.ollama_url,
                ollama_model=args.model,
                timeout=args.timeout,
            )
            reports.append(rep)
            progress.advance(task)

    save_outputs(reports, args.out)

    df = pd.read_csv(Path(args.out) / "report.csv")
    summary = (
        df.groupby(["item_id", "item_name", "verdict"])
        .size()
        .reset_index(name="count")
    )
    console.print("\n[bold]Summary (counts per item/verdict):[/bold]")
    console.print(summary.to_string(index=False))
    console.print(
        f"\nOutput salvati in: [bold]{args.out}[/bold] (report.jsonl, report.csv)"
    )


if __name__ == "__main__":
    main()
