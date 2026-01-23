"""
prisma_models.py

Definisce le strutture dati principali e alcune funzioni di utilità condivise
tra gli altri moduli del progetto PRISMA audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

Verdict = Literal["present", "absent", "unclear"]


# =============================================================================
# DATA CLASS PRINCIPALI
# =============================================================================


@dataclass
class Paper:
    """
    Rappresenta un singolo articolo scientifico caricato dai file JSON.

    Attributi:
        id: Identificativo interno (ID del portale o simile).
        doi: DOI dell'articolo, se disponibile.
        title: Titolo dell'articolo (testo pulito).
        abstract: Abstract dell'articolo (testo pulito, HTML rimosso).
        year: Anno di pubblicazione, se disponibile.
        portal: Nome del portale o sorgente, se disponibile.
        authors: Lista di autori (stringhe), se disponibile.
    """

    id: str
    doi: str
    title: str
    abstract: str
    year: Optional[int] = None
    portal: Optional[str] = None
    authors: Optional[List[str]] = None


@dataclass
class ItemResult:
    """
    Risultato di valutazione di un singolo item PRISMA per un paper.

    Attributi:
        item_id: Identificativo (es. "1", "4", "24") dell'item PRISMA.
        item_name: Nome descrittivo dell'item.
        verdict: Verdetto complessivo: "present", "absent" o "unclear".
        confidence: Confidenza (0.0–1.0) associata al verdetto.
        evidence: Lista di brevi estratti testuali usati come evidenza.
        notes: Note esplicative (euristiche o rationale dell'LLM).
        method: Metodo principale usato per il verdetto ("heuristic" o "llm").
    """

    item_id: str
    item_name: str
    verdict: Verdict
    confidence: float
    evidence: List[str]
    notes: str
    method: str


@dataclass
class PaperReport:
    """
    Report di valutazione PRISMA per un singolo paper.

    Attributi:
        paper_id: ID del paper (coerente con Paper.id).
        title: Titolo del paper.
        doi: DOI del paper.
        results: Lista di ItemResult, uno per ogni item PRISMA considerato.
    """

    paper_id: str
    title: str
    doi: str
    results: List[ItemResult]


@dataclass(frozen=True)
class PrismaItem:
    """
    Definizione di un item PRISMA considerato nell'audit.

    Attributi:
        item_id: Identificativo numerico o stringa (es. "1").
        name: Nome descrittivo (es. "Title identifies the report type").
        definition: Testo descrittivo dell'item (riassunto della checklist PRISMA).
        keywords: Lista di keyword di base per il matching euristico.
    """

    item_id: str
    name: str
    definition: str
    keywords: List[str]


# =============================================================================
# MODELLI Pydantic PER RISPOSTE LLM
# =============================================================================


class LLMJudgement(BaseModel):
    """
    Risposta dell'LLM relativa al giudizio su un item PRISMA.

    Attributi:
        verdict: Verdetto LLM ("present", "absent", "unclear").
        confidence: Confidenza LLM nel verdetto (0.0–1.0).
        evidence_quotes: Lista di brevi citazioni dal testo.
        rationale: Breve spiegazione del verdetto.
    """

    verdict: Verdict = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_quotes: List[str] = Field(default_factory=list)
    rationale: str = Field(...)


class KeywordExpansionResponse(BaseModel):
    """
    Risposta LLM per espansione delle keyword di un item PRISMA.

    Attributi:
        extra_keywords: Lista di keyword aggiuntive suggerite dall'LLM.
        notes: Breve spiegazione di cosa è stato aggiunto e perché.
    """

    extra_keywords: List[str] = Field(default_factory=list)
    notes: str = Field(default="")


class TextNormalizationResponse(BaseModel):
    """
    Risposta LLM per normalizzazione del testo di un paper.

    Attributi:
        normalized_text: Versione semplificata/normalizzata del testo (es. lower-case).
        important_phrases: Lista di frasi brevi considerate rilevanti ai fini PRISMA.
    """

    normalized_text: str = Field(default="")
    important_phrases: List[str] = Field(default_factory=list)


# =============================================================================
# FUNZIONI DI UTILITÀ CONDIVISE
# =============================================================================


def normalize_ws(s: str) -> str:
    """
    Normalizza gli spazi in una stringa, riducendo sequenze multiple a singoli spazi.

    Args:
        s: Stringa in input.

    Returns:
        Stringa con spazi normalizzati e trim agli estremi.
    """
    import re

    return re.sub(r"\s+", " ", (s or "")).strip()


def clean_html(text: str) -> str:
    """
    Rimuove HTML da una stringa e normalizza gli spazi.

    Args:
        text: Testo grezzo, eventualmente con markup HTML.

    Returns:
        Testo pulito, privo di markup e con spazi normalizzati.
    """
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    cleaned = soup.get_text(separator=" ", strip=True)
    return normalize_ws(cleaned)


def paper_text(p: Paper) -> str:
    """
    Costruisce il testo analizzabile per un paper (solo titolo + abstract).

    Args:
        p: Paper di interesse.

    Returns:
        Stringa contenente titolo e abstract con etichette.
    """
    return normalize_ws(f"TITLE: {p.title}\nABSTRACT: {p.abstract}")


def clip_quotes(snips: List[str], max_len: int = 420) -> List[str]:
    """
    Accorcia una lista di snippet testuali a una lunghezza massima
    e ne limita il numero.

    Args:
        snips: Lista di stringhe originali.
        max_len: Lunghezza massima di ciascuna stringa.

    Returns:
        Lista di al massimo 3 snippet accorciati.
    """
    out: List[str] = []
    for s in snips:
        s_norm = normalize_ws(s)
        if len(s_norm) > max_len:
            s_norm = s_norm[: max_len - 3] + "..."
        out.append(s_norm)
    return out[:3]


# =============================================================================
# DEFINIZIONE ITEM PRISMA (SUBSET)
# =============================================================================

ITEMS: List[PrismaItem] = [
    PrismaItem(
        item_id="1",
        name="Title identifies the report type",
        definition=(
            "Il titolo dovrebbe identificare il lavoro come systematic review "
            "o scoping review."
        ),
        keywords=[
            "systematic review",
            "scoping review",
            "review",
            "meta-analysis",
            "metaanalysis",
        ],
    ),
    PrismaItem(
        item_id="4",
        name="Objectives stated",
        definition=(
            "Dovrebbe esserci una dichiarazione esplicita dell'obiettivo/i "
            "o domanda/e della review."
        ),
        keywords=[
            "objective",
            "objectives",
            "aim",
            "aims",
            "purpose",
            "we aimed",
            "this study aims",
            "goal",
        ],
    ),
    PrismaItem(
        item_id="6",
        name="Information sources (databases) mentioned",
        definition=(
            "Dovrebbero essere menzionate fonti informative (es. database: "
            "PubMed, Scopus, Web of Science...)."
        ),
        keywords=[
            "pubmed",
            "medline",
            "embase",
            "scopus",
            "web of science",
            "cinahl",
            "cochrane",
            "ieee xplore",
            "psycinf",
        ],
    ),
    PrismaItem(
        item_id="24",
        name="Protocol/registration mentioned",
        definition=(
            "Dovrebbe essere indicato se esiste un protocollo/registrazione "
            "(es. PROSPERO) o dove trovarlo."
        ),
        keywords=[
            "prospero",
            "registered",
            "registration",
            "protocol",
            "osf",
            "preregistration",
            "pre-registered",
        ],
    ),
    PrismaItem(
        item_id="25",
        name="Funding/support mentioned",
        definition=(
            "Dovrebbero essere riportate fonti di finanziamento/supporto "
            "(a livello abstract spesso è assente)."
        ),
        keywords=[
            "funding",
            "funded",
            "supported by",
            "grant",
            "sponsor",
            "financial support",
        ],
    ),
    PrismaItem(
        item_id="26",
        name="Competing interests mentioned",
        definition=(
            "Dovrebbero essere riportati conflitti di interesse/competing interests "
            "(nell'abstract spesso è assente)."
        ),
        keywords=[
            "conflict of interest",
            "conflicts of interest",
            "competing interests",
            "no competing interests",
            "declare no conflict",
        ],
    ),
    PrismaItem(
        item_id="27",
        name="Data/code availability mentioned",
        definition=(
            "Dovrebbe essere indicata la disponibilità di dati/codice/materiali "
            "(spesso in sezioni finali)."
        ),
        keywords=[
            "data availability",
            "code availability",
            "github",
            "repository",
            "supplementary",
            "zenodo",
            "figshare",
            "osf",
        ],
    ),
]
