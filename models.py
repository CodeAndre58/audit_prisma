from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

Verdict = Literal["present", "absent", "unclear"]

# ----------------------------
# Models
# ----------------------------
@dataclass
class Paper:
    id: str
    doi: str
    title: str
    abstract: str
    year: Optional[int] = None
    portal: Optional[str] = None
    authors: Optional[List[str]] = None

@dataclass
class ItemResult:
    item_id: str
    item_name: str
    verdict: Verdict
    confidence: float
    evidence: List[str]
    notes: str
    method: str  # "heuristic" | "llm"

@dataclass
class PaperReport:
    paper_id: str
    title: str
    doi: str
    results: List[ItemResult]

# ----------------------------
# PRISMA items (subset abstract-level)
# ----------------------------
@dataclass(frozen=True)
class PrismaItem:
    item_id: str
    name: str
    definition: str
    keywords: List[str]

ITEMS: List[PrismaItem] = [
    PrismaItem(
        item_id="1",
        name="Title identifies the report type",
        definition="Il titolo dovrebbe identificare il lavoro come systematic review o scoping review.",
        keywords=["systematic review", "scoping review", "review", "meta-analysis", "metaanalysis"]
    ),
    PrismaItem(
        item_id="4",
        name="Objectives stated",
        definition="Dovrebbe esserci una dichiarazione esplicita dell'obiettivo/i o domanda/e della review.",
        keywords=["objective", "objectives", "aim", "aims", "purpose", "we aimed", "this study aims", "goal"]
    ),
    PrismaItem(
        item_id="6",
        name="Information sources (databases) mentioned",
        definition="Dovrebbero essere menzionate fonti informative (es. database: PubMed, Scopus, Web of Science...).",
        keywords=["pubmed", "medline", "embase", "scopus", "web of science", "cinahl", "cochrane", "ieee xplore", "psycinf"]
    ),
    PrismaItem(
        item_id="24",
        name="Protocol/registration mentioned",
        definition="Dovrebbe essere indicato se esiste un protocollo/registrazione (es. PROSPERO) o dove trovarlo.",
        keywords=["prospero", "registered", "registration", "protocol", "osf", "preregistration", "pre-registered"]
    ),
    PrismaItem(
        item_id="25",
        name="Funding/support mentioned",
        definition="Dovrebbero essere riportate fonti di finanziamento/supporto (a livello abstract spesso è assente).",
        keywords=["funding", "funded", "supported by", "grant", "sponsor", "financial support"]
    ),
    PrismaItem(
        item_id="26",
        name="Competing interests mentioned",
        definition="Dovrebbero essere riportati conflitti di interesse/competing interests (nell'abstract spesso è assente).",
        keywords=["conflict of interest", "conflicts of interest", "competing interests", "no competing interests", "declare no conflict"]
    ),
    PrismaItem(
        item_id="27",
        name="Data/code availability mentioned",
        definition="Dovrebbe essere indicata la disponibilità di dati/codice/materiali (spesso in sezioni finali).",
        keywords=["data availability", "code availability", "github", "repository", "supplementary", "zenodo", "figshare", "osf"]
    ),
]
