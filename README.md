# PRISMA Audit - Sistema Automatico di Validazione Conformità PRISMA

Sistema intelligente per l'audit automatico della conformità ai criteri PRISMA 2020 (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) mediante analisi di titolo e abstract di articoli scientifici. Utilizza metodi euristici e integrazione con modelli LLM (Ollama) per validazione precisa.

---

## Indice

1. [Panoramica](#panoramica)
2. [Requisiti di Sistema](#requisiti-di-sistema)
3. [Installazione](#installazione)
4. [Configurazione Variabili d'Ambiente](#configurazione-variabili-dambiente)
5. [Avvio Rapido (Quickstart)](#avvio-rapido-quickstart)
6. [Utilizzo e Esempi](#utilizzo-e-esempi)
7. [Tabella Parametri CLI](#tabella-parametri-cli)
8. [Integrazione Ollama](#integrazione-ollama)
9. [Algoritmo di Validazione](#algoritmo-di-validazione)
10. [Troubleshooting](#troubleshooting)

---

## Panoramica

Il progetto esegue un **audit automatico dei criteri PRISMA** su un set di articoli scientifici in formato JSON. Per ogni articolo analizza 7 item PRISMA fondamentali:

- **Item 1**: Titolo identifica il tipo di report
- **Item 4**: Obiettivi esplicitamente dichiarati
- **Item 6**: Fonti informative (database) citate
- **Item 24**: Protocollo/registrazione (es. PROSPERO)
- **Item 25**: Finanziamento/supporto menzionato
- **Item 26**: Conflitti di interesse dichiarati
- **Item 27**: Disponibilità dati/codice

Per ogni item, il sistema assegna un **verdetto** (present/absent/unclear) con **confidence score** (0.0-1.0) e **evidenze estratte dal testo**.

### Modalità di Validazione

- **Heuristic-only**: Ricerca basata su keyword matching (veloce, meno preciso)
- **LLM-enhanced**: Validazione aggiuntiva tramite Ollama con modello locale (più accurato)

---

## Requisiti di Sistema

### Software

- **Python**: 3.8+
- **pip**: Package manager Python
- **Ollama** (opzionale): Per modalità LLM
  - Download: https://ollama.ai
  - Installazione: segui guida ufficiale per il tuo OS

## Installazione

### 1. Clone o Preparazione Progetto

```bash
# Se da repository
git clone <repo-url>
cd uni_projects

# Oppure naviga alla cartella esistente
cd /home/andre/uni_projects
```

### 2. Creazione Virtual Environment

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installazione Dipendenze

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configurazione Variabili d'Ambiente

### File `.env` (Opzionale)

Crea un file `.env` nella radice del progetto per configurare parametri persistenti:

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_TIMEOUT=60
```

### Variabili Disponibili

| Variabile | Valore Default | Descrizione |
|-----------|----------------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL base del server Ollama |
| `OLLAMA_MODEL` | `llama3.1` | Nome modello Ollama disponibile |
| `OLLAMA_TIMEOUT` | `60` | Timeout in secondi per richieste LLM |

### Caricamento Automatico

Il file `.env` viene caricato automaticamente da `load_dotenv()` in `prisma_audit.py`.

---

## Avvio Rapido (Quickstart)

### Scenario 1: Analisi con Soli Metodi Euristici (Veloce)

```bash
python prisma_audit.py --input "input/*.json" --out output --llm none
```

**Risultato:** Report in `output/report.csv` e `output/report.jsonl` in ~5-10 secondi.

### Scenario 2: Analisi con Validazione LLM (Accurato)

Prerequisito: Ollama in esecuzione con modello `llama3.1`

```bash
# Terminal 1: Avvia Ollama
ollama serve

# Terminal 2: Esegui audit con LLM
python prisma_audit.py --input "input/*.json" --out output --llm ollama
```

**Risultato:** Report più accurato in 30-120 secondi (a seconda del numero di articoli).

### Scenario 3: File Singolo

```bash
python prisma_audit.py --input "input/test_input.json" --out output_test --llm none
```

---

## Utilizzo e Esempi

### Sintassi Generale

```bash
python prisma_audit.py [OPZIONI]
```

### Esempio 1: Multi-Pattern con LLM

```bash
python prisma_audit.py \
  --input "input/AI*.json" "input/Cryptography*.json" \
  --out output \
  --llm ollama \
  --model llama3.1 \
  --ollama-url http://localhost:11434 \
  --timeout 90
```

### Esempio 2: Elaborazione con Modello Diverso

```bash
python prisma_audit.py \
  --input "input/test_input.json" \
  --out custom_output \
  --llm ollama \
  --model mistral
```

*Nota:* Assicurati che il modello sia disponibile in Ollama (`ollama list`)

### Interpretazione Output

#### File: `output/report.csv`

```csv
paper_id,doi,title,item_id,item_name,verdict,confidence,method,evidence,notes
6646,10.5281/zenodo.13885454,Artificial Intelligence in Healthcare...,1,Title identifies...,present,0.80,heuristic,"systematic review" from title,Heuristic: keyword in title
6646,10.5281/zenodo.13885454,Artificial Intelligence in Healthcare...,4,Objectives stated,present,0.65,heuristic,This review evaluates the significant applications,Heuristic: keyword found
```

**Interpretazione verdetti:**
- **present** (0.5-1.0): Item chiaramente soddisfatto
- **absent** (0.4-0.7): Item non trovato nel testo
- **unclear** (0.4-0.6): Evidenza insufficiente (frequente negli abstract)

#### File: `output/report.jsonl`

```json
{"paper_id": "6646", "title": "Artificial Intelligence in Healthcare...", "doi": "10.5281/zenodo.13885454", "results": [{"item_id": "1", "item_name": "Title identifies...", "verdict": "present", "confidence": 0.80, "evidence": ["systematic review"], "notes": "...", "method": "heuristic"}]}
```

---

## Tabella Parametri CLI

| Parametro | Tipo | Default | Obbligatorio | Esempio | Descrizione |
|-----------|------|---------|------|---------|-------------|
| `--input` | glob string(s) | - | **SÌ** | `"input/*.json"` | Pattern glob per file JSON (1+ pattern) |
| `--out` | path | `output` | NO | `custom_output` | Cartella output per report |
| `--llm` | choice | `none` | NO | `ollama`, `none` | Modalità LLM: "none" (heuristic), "ollama" |
| `--model` | string | `llama3.1` | NO | `mistral`, `neural-chat` | Nome modello Ollama |
| `--ollama-url` | URL | `http://localhost:11434` | NO | `http://192.168.1.100:11434` | Base URL server Ollama (http/https) |
| `--timeout` | integer | `60` | NO | `120` | Timeout secondi per richieste LLM |

### Esempi di Pattern Glob

```bash
# Tutti i JSON nella cartella input
--input "input/*.json"

# JSON con prefisso specifico
--input "input/AI*.json"

# Multi-pattern
--input "input/AI*.json" "input/Cryptography*.json"

# Pattern ricorsivo (sottocartelle)
--input "input/**/*.json"
```

---

## Integrazione Ollama

### Setup Ollama

#### 1. Installazione

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
# Download da https://ollama.ai oppure
brew install ollama
```

**Windows:**
- Scarica installer da https://ollama.ai
- Esegui installer e segui procedure

#### 2. Avvio Server

```bash
# Linux/macOS
ollama serve

# Output atteso:
# time=2025-01-19T14:30:00Z level=INFO msg="Listening on 127.0.0.1:11434 (http)"
```

#### 3. Download Modello

```bash
# In nuovo terminal, mentre server è attivo
ollama pull llama3.1

# Verifica modelli disponibili
ollama list
```

**Modelli Consigliati:**
- `llama3.1` - Bilanciato (3.2GB)
- `mistral` - Veloce (4GB)
- `neural-chat` - Optimizzato chat (4GB)

### Errori Comuni e Soluzioni

#### ❌ Connection refused (http://localhost:11434)

```
requests.exceptions.ConnectionError: Connection refused
```

**Soluzione:**
```bash
# 1. Verifica server in esecuzione
ps aux | grep ollama

# 2. Avvia server se non attivo
ollama serve &

# 3. Testa connessione
curl http://localhost:11434/api/tags
```

#### ❌ Model not found

```
Error: model not found
```

**Soluzione:**
```bash
# Verifica modelli disponibili
ollama list

# Se mancante, scarica
ollama pull llama3.1

# Usa nome esatto nel comando
python prisma_audit.py --model llama3.1 ...
```

#### ❌ Timeout durante elaborazione

```
requests.exceptions.Timeout: HTTPConnectionPool(...) Read timed out
```

**Soluzione:**
```bash
# Aumenta timeout (default 60 secondi)
python prisma_audit.py --timeout 120 ...

# Oppure controlla risorse Ollama
ollama ps

# Se necessario, riavvia server
pkill ollama
ollama serve &
```

#### ❌ Modello non risponde in JSON valido

```
LLM output non parsabile, fallback heuristic. Errore: JSONDecodeError
```

**Comportamento:** Sistema automaticamente fallback a metodo heuristic con confidence ridotta.

**Prevenzione:**
- Usa modelli consolidati (llama3.1, mistral)
- Imposta `temperature: 0` (già nel codice) per output deterministico
- Verifica prompt in `build_judge_prompt()`

---

## Algoritmo di Validazione

### Architettura Generale

L'algoritmo opera in **due fasi**:

```
[INPUT: Paper (title + abstract)]
         ↓
    [HEURISTIC PHASE]
         ↓
[LLM PHASE (opzionale)]
         ↓
[OUTPUT: ItemResult con verdict+confidence]
```

### Fase 1: Validazione Euristica (SEMPRE)

#### Input

```python
# Input ricevuto
item = PrismaItem(
    item_id="6",
    name="Information sources (databases) mentioned",
    definition="Dovrebbero essere menzionate fonti informative...",
    keywords=["pubmed", "medline", "scopus", "web of science", ...]
)

paper_text = """
TITLE: Systematic Review of AI Applications in Healthcare
ABSTRACT: We conducted a systematic review by searching PubMed and Scopus
databases from 2020 to 2024. Study selection was performed independently...
"""
```

#### Regole di Validazione

```python
# heuristic_check() in prisma_audit.py (lines 85-120)

def heuristic_check(item: PrismaItem, text: str) -> ItemResult:
    """
    REGOLE:
    1. Normalizzazione testo: lowercase, whitespace cleanup
    2. Keyword matching: ricerca keyword in testo
    3. Estrazione evidence: snippet di 120 char prima + 200 char dopo match
    4. Assegnazione verdetto:
       - Logica Item 1 (Title): più rigida, require keyword in titolo
       - Item 24,25,26,27: "unclear" se non trovato (spesso in full-text)
       - Altrimenti: "present" se hits, "absent" altrimenti
    5. Calcolo confidence: basato su presence/absence di keyword
    """
```

#### Passaggi Dettagliati

**Step 1: Normalizzazione Testo**

```python
text_lower = paper_text.lower()
# Risultato: lowercase + whitespace uniforme
# "TITLE: Systematic Review... PubMed and Scopus..."
# → "title: systematic review... pubmed and scopus..."
```

**Step 2: Ricerca Keyword**

```python
keywords = item.keywords  # ["pubmed", "medline", "scopus", ...]
hits = [kw for kw in keywords if kw.lower() in text_lower]
# Risultato (per l'esempio): hits = ["pubmed", "scopus"]
```

**Step 3: Estrazione Evidence**

```python
# Per ogni keyword trovato, estrai context snippet
for kw in hits[:2]:  # Max 2 keyword
    m = re.search(re.escape(kw), text_lower)
    if m:
        start = max(0, m.start() - 120)
        end = min(len(text), m.end() + 200)
        snippet = paper_text[start:end]  # Usa testo originale (maiuscole)
        evidence.append(snippet)

# Risultato:
# [
#   "...our systematic review by searching PubMed and Scopus databases...",
#   "...Scopus was selected as primary source for this..."
# ]
```

**Step 4: Assegnazione Verdetto (Logica Speciale per Item)**

```python
if item.item_id == "1":  # Item Title - logica più rigida
    in_title = any(kw in (text.split("\n", 1)[0].lower()) 
                   for kw in item.keywords)
    verdict = "present" if in_title else ("unclear" if hits else "absent")
    confidence = 0.80 if in_title else (0.55 if hits else 0.65)
    
elif item.item_id in {"24", "25", "26", "27"}:  # Item senza abstract
    verdict = "present" if hits else "unclear"
    confidence = 0.65 if hits else 0.50
    # "unclear" perché info spesso in full-text, non abstract
    
else:  # Item standard
    verdict = "present" if hits else "absent"
    confidence = 0.65 if hits else 0.55
```

**Step 5: Clipping Evidence**

```python
def clip_quotes(evidence_list, max_len=420):
    # Limita ogni quote a 420 char, max 3 quote
    # Previene output eccessivo
```

#### Output Fase Euristica

```python
ItemResult(
    item_id="6",
    item_name="Information sources (databases) mentioned",
    verdict="present",  # ← Assegnato dalle regole
    confidence=0.65,    # ← Assegnato dalle regole
    evidence=[
        "...conducting a systematic review by searching PubMed and Scopus databases...",
        "...identified through manual screening of Scopus results..."
    ],
    notes="Heuristic: trovate keyword ['pubmed', 'scopus'] nel testo.",
    method="heuristic"
)
```

---

### Fase 2: Validazione LLM (OPZIONALE, se --llm ollama)

#### Precondizione

- Server Ollama attivo
- Modello disponibile
- Heuristic result già calcolato

#### Input a LLM

```python
prompt = build_judge_prompt(item, paper, heuristic_result)
# Contenuto:
"""
ITEM:
- id: 6
- name: Information sources (databases) mentioned
- definition: Dovrebbero essere menzionate fonti informative (es. database: PubMed, Scopus, Web of Science...)

PAPER TEXT (title + abstract only):
TITLE: Systematic Review of AI Applications in Healthcare
ABSTRACT: We conducted a systematic review by searching PubMed and Scopus...

HEURISTIC PRE-CHECK:
- verdict: present
- confidence: 0.65
- evidence: ["...PubMed and Scopus databases..."]
- notes: "Heuristic: trovate keyword..."

TASK:
Decide whether the ITEM is satisfied using ONLY the PAPER TEXT above.
If the evidence is insufficient (common with abstract-only), return verdict='unclear'.

Return ONLY this JSON (no markdown):
{
  "verdict": "present|absent|unclear",
  "confidence": 0.0,
  "evidence_quotes": ["short quote 1"],
  "rationale": "1-3 sentences"
}
"""
```

#### Regole LLM

```python
# build_judge_prompt() in prisma_audit.py

system_message = """You are a strict PRISMA compliance auditor. 
Use only the provided text. Output ONLY valid JSON."""

# Temperatura: 0 (deterministico)
# Questo forza il modello a dare risposte coerenti
```

## Riferimenti Aggiuntivi

### PRISMA 2020 Checklist

- **Documento ufficiale**: https://www.prisma-statement.org/
- **Item 1-27**: Descrizioni complete nel source ([models.py](models.py))

### Formati Output

- **CSV**: Facile importazione in Excel/Pandas, una riga per item per paper
- **JSONL**: Facilita streaming processing, una riga = un paper completo

### Librerie Utilizzate

| Libreria | Ruolo |
|----------|-------|
| `pydantic` | Validazione modelli dati |
| `requests` | HTTP client per Ollama |
| `pandas` | Processing dati, output CSV |
| `rich` | Terminal UI (progress bar, colori) |
| `bs4` + `lxml` | Parsing HTML abstract |
| `python-dotenv` | Load .env config |

---

## Conclusioni

Questo sistema fornisce un **audit robusto e scalabile** della conformità PRISMA. La combinazione di metodi euristici (veloci) e LLM (accurati) permette flessibilità in base alle esigenze di velocità vs. precisione.

**Versione Documento:** 1.0  
**Data:** 19 Gennaio 2025  
**Autore:** Andrea Di Cristina 
**Email** andrea.dicristina63@gmail.com