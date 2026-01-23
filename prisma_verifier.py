"""
prisma_verifier.py

Contiene tutta la logica di verifica PRISMA:
- matching euristico tramite keyword (base + espanse via LLM),
- chiamate LLM (Ollama/OpenAI) per giudizio, espansione keyword,
  normalizzazione testo,
- orchestrazione dell'audit per singolo paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from pydantic import ValidationError
from rich.console import Console

from models import (
    ITEMS,
    ItemResult,
    KeywordExpansionResponse,
    LLMJudgement,
    Paper,
    PaperReport,
    PrismaItem,
    TextNormalizationResponse,
    Verdict,
    clip_quotes,
    normalize_ws,
    paper_text,
)

console = Console()


# =============================================================================
# HELPER GENERALE PER JSON NELLA RISPOSTA LLM
# =============================================================================


def extract_json_object(s: str) -> str:
    """
    Estrae un oggetto JSON da una stringa che può contenere testo extra.

    Args:
        s: Stringa potenzialmente contenente un oggetto JSON.

    Returns:
        Sottostringa che rappresenta l'oggetto JSON più esterno,
        o la stringa originale se non vengono trovate graffe.
    """
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def ollama_chat_json(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Esegue una chiamata a Ollama (modalità chat) e restituisce un dict
    parsato da JSON dal contenuto del messaggio.

    L'LLM deve restituire un oggetto JSON nel campo content.

    Args:
        base_url: URL base dell'API Ollama (es. "http://localhost:11434").
        model: Nome del modello da usare (es. "qwen2.5:7b").
        system_prompt: Prompt di ruolo per l'LLM.
        user_prompt: Prompt utente con il task.
        timeout: Timeout in secondi per la richiesta HTTP.

    Returns:
        Dizionario risultante dal parsing del contenuto JSON dell'LLM.

    Raises:
        requests.HTTPError: In caso di errore nella risposta HTTP.
        json.JSONDecodeError: Se il contenuto non è JSON valido.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "") or ""
    raw = extract_json_object(content)
    return json.loads(raw)


# =============================================================================
# LLM CLIENT (INTERFACCIA + IMPLEMENTAZIONI)
# =============================================================================


class LLMClient:
    """
    Interfaccia astratta per un client LLM che effettua il giudizio
    su un item PRISMA per un paper.
    """

    def judge(
        self, item: PrismaItem, paper: Paper, heuristic: ItemResult
    ) -> ItemResult:
        """
        Esegue il giudizio LLM per un item specifico.

        Args:
            item: Item PRISMA da valutare.
            paper: Paper da esaminare.
            heuristic: Risultato euristico calcolato in precedenza.

        Returns:
            ItemResult aggiornato con le informazioni dell'LLM.
        """
        raise NotImplementedError


class OllamaClient(LLMClient):
    """
    Implementazione di LLMClient che utilizza un server Ollama locale
    via API HTTP.
    """

    def __init__(self, base_url: str, model: str, timeout: int = 60):
        """
        Inizializza un client Ollama.

        Args:
            base_url: URL base dell'API Ollama.
            model: Nome del modello da usare.
            timeout: Timeout per le richieste HTTP.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def judge(self, item: PrismaItem, paper: Paper, heuristic: ItemResult) -> ItemResult:
        """
        Esegue il giudizio LLM usando Ollama per un item specifico.

        Args:
            item: Item PRISMA da valutare.
            paper: Paper da esaminare.
            heuristic: Risultato euristico pre-esistente.

        Returns:
            ItemResult con verdetto dell'LLM o fallback euristico.
        """
        prompt = build_judge_prompt(item, paper, heuristic)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict PRISMA compliance auditor. "
                        "Use only the provided text. Output ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }

        r = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "") or ""
        return parse_llm_output(item, heuristic, content)


class OpenAIChatClient(LLMClient):
    """
    Implementazione di LLMClient usando le API OpenAI.

    Nota: opzionale, richiede:
        - pip install openai
        - variabile d'ambiente OPENAI_API_KEY impostata.
    """

    def __init__(self, model: str):
        """
        Inizializza il client OpenAI.

        Args:
            model: Nome del modello OpenAI (es. "gpt-4o-mini").

        Raises:
            ImportError: Se la libreria openai non è installata.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "La libreria 'openai' non è installata. "
                "Installa con: pip install openai"
            ) from e
        self.client = OpenAI()
        self.model = model

    def judge(self, item: PrismaItem, paper: Paper, heuristic: ItemResult) -> ItemResult:
        """
        Esegue il giudizio LLM usando OpenAI per un item specifico.

        Args:
            item: Item PRISMA da valutare.
            paper: Paper da esaminare.
            heuristic: Risultato euristico pre-esistente.

        Returns:
            ItemResult con verdetto dell'LLM o fallback euristico.
        """
        prompt = build_judge_prompt(item, paper, heuristic)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict PRISMA compliance auditor. "
                        "Use only the provided text. Output ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        return parse_llm_output(item, heuristic, content)


# =============================================================================
# PROMPT E PARSING RISPOSTA LLM PER GIUDIZIO ITEM
# =============================================================================


def build_judge_prompt(item: PrismaItem, paper: Paper, heuristic: ItemResult) -> str:
    """
    Costruisce il prompt testuale per il giudizio LLM su un item specifico.

    Args:
        item: Item PRISMA in esame.
        paper: Paper di riferimento.
        heuristic: Risultato euristico pre-esistente.

    Returns:
        Stringa da inviare come prompt utente all'LLM.
    """
    txt = paper_text(paper)
    return (
        f"ITEM:\n- id: {item.item_id}\n- name: {item.name}\n- definition: {item.definition}\n\n"
        f"PAPER TEXT (title + abstract only):\n{txt}\n\n"
        f"HEURISTIC PRE-CHECK:\n"
        f"- verdict: {heuristic.verdict}\n- confidence: {heuristic.confidence}\n"
        f"- evidence: {heuristic.evidence}\n- notes: {heuristic.notes}\n\n"
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


def parse_llm_output(
    item: PrismaItem, heuristic: ItemResult, content: str
) -> ItemResult:
    """
    Parsea il contenuto restituito dall'LLM e lo converte in ItemResult.

    In caso di errore di parsing o validazione, ritorna un ItemResult
    basato sul risultato euristico (fallback).

    Args:
        item: Item PRISMA relativo al giudizio.
        heuristic: Risultato euristico pre-esistente.
        content: Stringa contenente la risposta del modello.

    Returns:
        ItemResult con verdetto LLM o fallback euristico.
    """
    raw = extract_json_object(content)
    try:
        obj = json.loads(raw)
        parsed = LLMJudgement(**obj)
        evidence = clip_quotes(parsed.evidence_quotes)
        notes = f"LLM rationale: {parsed.rationale}"
        return ItemResult(
            item.item_id,
            item.name,
            parsed.verdict,
            float(parsed.confidence),
            evidence,
            notes,
            method="llm",
        )
    except (json.JSONDecodeError, ValidationError) as e:
        return ItemResult(
            item.item_id,
            item.name,
            heuristic.verdict,
            max(0.30, heuristic.confidence - 0.10),
            heuristic.evidence,
            f"LLM output non parsabile, fallback heuristic. Errore: {type(e).__name__}",
            method="heuristic",
        )


# =============================================================================
# ESPANSIONE KEYWORD VIA LLM
# =============================================================================


def expand_item_keywords_with_llm(
    item: PrismaItem,
    base_url: str,
    model: str,
    timeout: int = 60,
) -> KeywordExpansionResponse:
    """
    Usa l'LLM (Ollama) per generare keyword aggiuntive per un item PRISMA.

    Args:
        item: Item PRISMA per cui espandere la lista di keyword.
        base_url: URL base di Ollama.
        model: Nome del modello da usare.
        timeout: Timeout per la richiesta HTTP.

    Returns:
        Istanza KeywordExpansionResponse con extra_keywords e note.
    """
    system_prompt = (
        "You are helping to build robust keyword lists for detecting PRISMA checklist items "
        "in scientific titles and abstracts. You respond ONLY with valid JSON."
    )

    user_prompt = (
        f"ITEM:\n"
        f"- id: {item.item_id}\n"
        f"- name: {item.name}\n"
        f"- definition: {item.definition}\n\n"
        f"Current base keywords (English, some already good):\n"
        f"{item.keywords}\n\n"
        "TASK:\n"
        "- Propose additional keywords, synonyms, variants, acronyms, spelling variants,\n"
        "- Include Italian/English variants if meaningful.\n"
        "- Focus on how these concepts may appear in titles and abstracts of medical papers.\n"
        "- Avoid overly generic words like 'study', 'paper', 'research'.\n"
        "- Return ONLY this JSON object:\n"
        "{\n"
        "  \"extra_keywords\": [\"keyword 1\", \"keyword 2\", \"keyword 3\"],\n"
        "  \"notes\": \"short explanation of what you added\"\n"
        "}\n"
    )

    data = ollama_chat_json(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=timeout,
    )
    return KeywordExpansionResponse(**data)


def build_expanded_keywords(
    base_url: str,
    model: str,
    timeout: int = 60,
    cache_path: str = "expanded_keywords.json",
) -> Dict[str, List[str]]:
    """
    Costruisce (o carica se già esistente) una mappa di keyword espanse
    per ogni item PRISMA, usando l'LLM per generare sinonimi/varianti.

    Args:
        base_url: URL base di Ollama.
        model: Nome del modello da usare.
        timeout: Timeout per le richieste HTTP.
        cache_path: Percorso del file JSON di cache delle keyword espanse.

    Returns:
        Dizionario {item_id: [lista di keyword espanse + base]}.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    console.print(
        "[bold yellow]Espansione keyword PRISMA con LLM (una tantum)...[/bold yellow]"
    )
    all_keywords: Dict[str, List[str]] = {}

    for item in ITEMS:
        console.print(f"  → Item {item.item_id} - {item.name}")
        resp = expand_item_keywords_with_llm(item, base_url, model, timeout=timeout)
        merged = item.keywords + [
            kw for kw in resp.extra_keywords if kw not in item.keywords
        ]
        merged_norm = sorted(
            {normalize_ws(kw.lower()) for kw in merged if kw.strip()}
        )
        all_keywords[item.item_id] = merged_norm

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_keywords, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Keyword espanse salvate in {cache_file}[/green]")
    return all_keywords


# =============================================================================
# NORMALIZZAZIONE TESTO VIA LLM
# =============================================================================


def normalize_text_with_llm_for_matching(
    paper: Paper,
    base_url: str,
    model: str,
    timeout: int = 60,
) -> TextNormalizationResponse:
    """
    Usa l'LLM per creare una versione normalizzata del testo (TITLE+ABSTRACT)
    e per estrarre alcune frasi chiave rilevanti.

    Args:
        paper: Paper da normalizzare.
        base_url: URL base di Ollama.
        model: Nome del modello da usare.
        timeout: Timeout per la richiesta HTTP.

    Returns:
        Istanza TextNormalizationResponse con normalized_text e important_phrases.
    """
    text = paper_text(paper)

    system_prompt = (
        "You are helping with keyword-based detection of PRISMA items in scientific titles and abstracts.\n"
        "You will rewrite the text into a simplified, normalized form that is easier to match with keywords.\n"
        "Always respond ONLY with valid JSON."
    )

    user_prompt = (
        "Here is the TITLE and ABSTRACT of a paper:\n"
        "-----------------\n"
        f"{text}\n"
        "-----------------\n\n"
        "TASK:\n"
        "- Produce a 'normalized_text' where you:\n"
        "  * lower-case everything,\n"
        "  * expand obvious abbreviations when you can (e.g. 'SR' -> 'systematic review'),\n"
        "  * unify synonyms into standard forms (e.g. 'systematic literature review' -> 'systematic review'),\n"
        "  * translate Italian phrases to English when they express key review concepts.\n"
        "- Produce also a list 'important_phrases' of short key phrases relevant for methods, objectives, databases,\n"
        "  protocol/registration, funding, competing interests, and data/code availability.\n\n"
        "Return ONLY this JSON object:\n"
        "{\n"
        "  \"normalized_text\": \"...\",\n"
        "  \"important_phrases\": [\"phrase 1\", \"phrase 2\"]\n"
        "}\n"
    )

    data = ollama_chat_json(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=timeout,
    )
    return TextNormalizationResponse(**data)


# =============================================================================
# EURISTICA DI CHECK PRISMA
# =============================================================================


def heuristic_check(
    item: PrismaItem,
    text: str,
    expanded_kw_map: Optional[Dict[str, List[str]]] = None,
) -> ItemResult:
    """
    Esegue una verifica euristica per un item PRISMA basandosi su keyword.

    Usa la mappa di keyword espanse (se fornita) o le keyword di base dell'item.

    Args:
        item: Item PRISMA da valutare.
        text: Testo su cui effettuare il matching (es. titolo+abstract, eventualmente arricchito).
        expanded_kw_map: Mappa opzionale {item_id: [keyword]} generata dall'LLM.

    Returns:
        ItemResult con verdetto euristico, confidenza e snippet di evidenza.
    """
    if expanded_kw_map and item.item_id in expanded_kw_map:
        keywords = expanded_kw_map[item.item_id]
    else:
        keywords = item.keywords

    t = text.lower()
    hits = [kw for kw in keywords if kw.lower() in t]
    evidence: List[str] = []

    if hits:
        for kw in hits[:2]:
            pattern = re.escape(kw)
            m = re.search(pattern, t)
            if not m:
                continue
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 200)
            snippet = text[start:end]
            evidence.append(snippet)

    if item.item_id == "1":
        first_line = text.split("\n", 1)[0].lower()
        in_title = any(kw in first_line for kw in keywords)
        verdict: Verdict = "present" if in_title else ("unclear" if hits else "absent")
        conf = 0.80 if in_title else (0.55 if hits else 0.65)
        notes = (
            "Heuristic: controllo keyword nel titolo (forte) e nel testo complessivo (debole)."
        )
        return ItemResult(
            item.item_id,
            item.name,
            verdict,
            conf,
            clip_quotes(evidence),
            notes,
            method="heuristic",
        )

    if hits:
        verdict = "present"
        conf = 0.65
        notes = (
            f"Heuristic: trovate keyword {hits[:4]} nel testo (titolo+abstract normalizzato)."
        )
    else:
        verdict = "absent"
        conf = 0.55
        notes = (
            "Heuristic: nessuna keyword trovata (ma l'abstract potrebbe non contenere la sezione richiesta)."
        )

    if item.item_id in {"24", "25", "26", "27"} and not hits:
        verdict = "unclear"
        conf = 0.50
        notes = (
            "Heuristic: non trovato nell'abstract; spesso appare solo nel full-text (quindi 'unclear')."
        )

    return ItemResult(
        item.item_id,
        item.name,
        verdict,
        conf,
        clip_quotes(evidence),
        notes,
        method="heuristic",
    )


# =============================================================================
# AUDIT DI UN SINGOLO PAPER
# =============================================================================


def audit_paper(
    p: Paper,
    llm: Optional[LLMClient],
    expanded_kw_map: Optional[Dict[str, List[str]]] = None,
    normalize_with_llm: bool = False,
    ollama_base_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    timeout: int = 60,
) -> PaperReport:
    """
    Esegue l'audit PRISMA (euristico + LLM opzionale) per un singolo paper.

    Args:
        p: Paper da valutare.
        llm: Client LLM per il giudizio finale sugli item, o None per usare solo euristiche.
        expanded_kw_map: Mappa di keyword espanse per ogni item (opzionale).
        normalize_with_llm: Se True, chiede all'LLM una versione normalizzata del testo.
        ollama_base_url: URL di Ollama, necessario se normalize_with_llm è True.
        ollama_model: Modello da usare per la normalizzazione, se richiesta.
        timeout: Timeout per le chiamate all'LLM (normalizzazione).

    Returns:
        PaperReport contenente tutti gli ItemResult per il paper.
    """
    base_text = paper_text(p)
    text_for_heuristics = base_text

    if normalize_with_llm and ollama_base_url and ollama_model:
        try:
            norm = normalize_text_with_llm_for_matching(
                p,
                base_url=ollama_base_url,
                model=ollama_model,
                timeout=timeout,
            )
            extra_block = "\n\nNORMALIZED_TEXT:\n" + norm.normalized_text
            if norm.important_phrases:
                extra_block += "\nIMPORTANT_PHRASES:\n" + "; ".join(
                    norm.important_phrases
                )
            text_for_heuristics = base_text + extra_block
        except Exception as e:
            console.print(
                f"[yellow]Normalizzazione LLM fallita per paper {p.id}: {e}[/yellow]"
            )
            text_for_heuristics = base_text

    results: List[ItemResult] = []
    for item in ITEMS:
        h = heuristic_check(item, text_for_heuristics, expanded_kw_map=expanded_kw_map)
        if llm is None:
            results.append(h)
        else:
            results.append(llm.judge(item, p, h))

    return PaperReport(paper_id=p.id, title=p.title, doi=p.doi, results=results)
