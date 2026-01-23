from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

Verdict = Literal["present", "absent", "unclear"]

# ----------------------------
# Models
# ----------------------------
@dataclass
class Paper:
    """
    Classe che rappresenta un paper/articolo scientifico.
    
    Attributi:
        id (str): Identificativo univoco del paper
        doi (str): Digital Object Identifier
        title (str): Titolo del paper
        abstract (str): Abstract/sommario del paper
        year (Optional[int]): Anno di pubblicazione (opzionale)
        portal (Optional[str]): Portale/database da cui proviene (es. PubMed, Scopus)
        authors (Optional[List[str]]): Lista degli autori (opzionale)
    
    Descrizione:
        Rappresenta un articolo scientifico con metadati fondamentali e abstract.
        Viene utilizzato dal sistema di audit PRISMA per analizzare il compliance
        degli articoli rispetto agli item della checklist PRISMA.
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
    Classe che rappresenta il risultato dell'audit di un singolo item PRISMA.
    
    Attributi:
        item_id (str): Identificativo dell'item PRISMA (es. "1", "4", "6")
        item_name (str): Nome descrittivo dell'item (es. "Title identifies the report type")
        verdict (Verdict): Verdetto del check ('present', 'absent', 'unclear')
        confidence (float): Livello di confidenza del verdetto (0.0-1.0)
        evidence (List[str]): Lista di citazioni/snippet dal testo come evidenza
        notes (str): Note aggiuntive con spiegazione del verdetto
        method (str): Metodo usato per il check ('heuristic' o 'llm')
    
    Descrizione:
        Rappresenta il risultato dell'analisi di un item PRISMA per un paper.
        Contiene il verdetto, la confidenza, le evidenze estratte e il metodo
        utilizzato (euristica o LLM).
    """
    item_id: str
    item_name: str
    verdict: Verdict
    confidence: float
    evidence: List[str]
    notes: str
    method: str  # "heuristic" | "llm"

@dataclass
class PaperReport:
    """
    Classe che rappresenta il report completo di audit di un paper.
    
    Attributi:
        paper_id (str): Identificativo univoco del paper
        title (str): Titolo del paper
        doi (str): Digital Object Identifier del paper
        results (List[ItemResult]): Lista di risultati per ciascun item PRISMA verificato
    
    Descrizione:
        Aggregazione di tutti i risultati dell'audit PRISMA per un singolo paper.
        Contiene metadati del paper e tutti gli ItemResult per gli item analizzati.
        Viene salvato in formato JSONL (una riga per paper) e CSV (con una riga
        per ciascun item per paper).
    """
    paper_id: str
    title: str
    doi: str
    results: List[ItemResult]

# ----------------------------
# PRISMA items (subset abstract-level)
# ----------------------------
@dataclass(frozen=True)
class PrismaItem:
    """
    Classe che rappresenta un item della checklist PRISMA.
    
    Attributi:
        item_id (str): Identificativo univoco dell'item (es. "1", "4", "6", "24", "25", "26", "27")
        name (str): Nome/titolo dell'item (es. "Title identifies the report type")
        definition (str): Definizione estesa dell'item in italiano
        keywords (List[str]): Lista di parole chiave usate per il matching euristico
    
    Descrizione:
        Rappresenta un singolo item della checklist PRISMA 2020 (subset a livello abstract).
        È immutabile (frozen=True) poiché rappresenta la definizione statica degli item.
        Viene utilizzato per parametrizzare i check euristici e LLM durante l'audit.
        La lista keywords è usata per il matching keyword-based durante il controllo euristico.
    """
    item_id: str
    name: str
    definition: str
    keywords: List[str]

# ----------------------------
# PRISMA items (subset abstract-level)
# ----------------------------
# Lista di item PRISMA da auditare (subset abstract-level).
# Contiene gli item principali della checklist PRISMA 2020 che possono essere
# verificati a livello di abstract (titolo + abstract).
# Item omessi: quelli che richiedono il full-text come Methods, Results, Discussion.
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
