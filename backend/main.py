"""
FinSight AI Backend - SEC 10-K/10-Q Filing Analysis System

FastAPI backend with:
- Async background jobs (POST /analyze_filing/ returns job_id immediately)
- SSE streaming Q&A with classify → retrieve/fetch-live → generate pipeline
- Multi-turn conversation memory per session_id
- In-memory analysis cache (populated on first request per ticker per session)
- FAISS indexes persisted to HF Dataset repo, restored on startup
- Dual LLM provider: Ollama (self-hosted) or NVIDIA NIM (cloud API)
- LangSmith observability via env vars
"""

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, Literal
from uuid import uuid4

import requests
import yfinance as yf
from edgar import Company, set_identity
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import hf_store
from faiss_manager import faiss_manager
from langchain_pipeline import SYSTEM_PROMPT, get_llm, get_provider_name
from plots import get_company_filings_data, plot_balance_sheet, plot_cash_flow, plot_revenue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["processing", "complete", "error"]
    progress: str
    result: dict[str, Any] | None = None
    error: str | None = None


class CompanySearchResult(BaseModel):
    ticker: str
    cik: str
    title: str


class AgentResponse(BaseModel):
    answer: str
    cited_sections: list[str]
    source: Literal["document", "realtime", "hybrid"]


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

jobs: dict[str, dict[str, Any]] = {}
analysis_cache: dict[str, dict[str, Any]] = {}
active_jobs: set = set()
ticker_locks: dict[str, asyncio.Lock] = {}


# Per-session conversation history: session_id -> list of (role, content) tuples
session_history: dict[str, list[dict[str, str]]] = {}
SESSION_CAP = 50


# ---------------------------------------------------------------------------
# yfinance cash flow supplement for Financial Performance section
# ---------------------------------------------------------------------------


def get_yfinance_dividends(ticker: str) -> str:
    """
    Fetch annual dividend per share from yfinance to supplement Strategic Initiatives
    when the filing's dividend note chunk is not retrieved by FAISS.
    Only includes years with 4 quarterly payments to avoid partial-year distortion.
    """
    try:
        t = yf.Ticker(ticker)
        divs = t.dividends
        if divs is None or divs.empty:
            return ""
        annual_counts = divs.groupby(divs.index.year).count()
        annual_sums = divs.groupby(divs.index.year).sum()
        # Only include years with at least 3 payments (guards against partial years)
        complete_years = annual_counts[annual_counts >= 3].index
        filtered = annual_sums[annual_sums.index.isin(complete_years)]
        if filtered.empty:
            return ""
        recent = filtered.tail(3).iloc[::-1]
        lines = ["Dividend Data — reproduce each line below exactly:"]
        for year, amount in recent.items():
            lines.append(f"  - **{year}:** ${amount:.2f} per share (annual dividends paid)")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"yfinance dividends fetch failed for {ticker}: {e}")
        return ""


def get_yfinance_rd(ticker: str) -> str:
    """
    Fetch R&D expense from yfinance income statement.
    Supplements Strategic Initiatives when FAISS retrieves the total operating expenses
    row instead of the R&D sub-line.
    """
    try:
        t = yf.Ticker(ticker)
        inc = t.income_stmt
        if inc is None or inc.empty:
            return ""
        rd_keys = ["Research And Development", "Research & Development", "ResearchAndDevelopment"]
        rd_row = next((inc.loc[k] for k in rd_keys if k in inc.index), None)
        if rd_row is None:
            return ""
        lines = [
            "R&D Expense (from income statement) — use these figures for Research and Development expense:"
        ]
        for col in list(inc.columns)[:3]:
            val = rd_row[col]
            if val != val:  # NaN
                continue
            lines.append(
                f"  - **{col.year}:** R&D expense ${abs(int(val)) / 1e9:.3f}B (${abs(int(val)) / 1e6:.0f}M)"
            )
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        logger.warning(f"yfinance R&D fetch failed for {ticker}: {e}")
        return ""


def get_yfinance_income_stmt(ticker: str) -> str:
    """
    Fetch key income statement metrics from yfinance, pre-converted to correct B/M units.
    Covers: Revenue, Gross Profit, Operating Income, Operating Expenses, Net Income, EPS.

    All rows are included — the denomination arithmetic is done in Python, not by the LLM.
    yfinance fiscal-year alignment may differ slightly for non-calendar fiscal years (e.g.
    Jan 31 year-end), but is still far more accurate than the LLM misreading filing units.
    """
    try:
        t = yf.Ticker(ticker)
        inc = t.income_stmt
        if inc is None or inc.empty:
            return ""

        def _fmt_dollars(val) -> str | None:
            try:
                v = float(val)
            except (TypeError, ValueError):
                return None
            if v != v:  # NaN
                return None
            sign = "-" if v < 0 else ""
            mag = abs(v)
            if mag >= 1e9:
                return f"{sign}${mag / 1e9:.3f}B"
            return f"{sign}${mag / 1e6:.0f}M"

        def _get_row(keys):
            return next((inc.loc[k] for k in keys if k in inc.index), None)

        rev_row = _get_row(["Total Revenue", "Revenue"])
        gp_row = _get_row(["Gross Profit"])
        oi_row = _get_row(["Operating Income", "EBIT"])
        opex_row = _get_row(["Operating Expense"])
        ni_row = _get_row(["Net Income", "Net Income Common Stockholders"])
        # EPS intentionally excluded: yfinance fiscal-year alignment differs for non-calendar
        # fiscal year companies (e.g. CRWD Jan 31 year-end), causing EPS/Net Income mismatch.
        # EPS is per-share (already in dollars) so denomination errors don't apply — FAISS is safe.

        cols = list(inc.columns)[:3]
        lines = []

        for col in cols:
            year = col.year
            parts = []
            for label, row in [
                ("Revenue", rev_row),
                ("Gross Profit", gp_row),
                ("Operating Income", oi_row),
                ("Operating Expenses", opex_row),
                ("Net Income", ni_row),
            ]:
                if row is None:
                    continue
                v = _fmt_dollars(row[col])
                if v:
                    parts.append(f"{label}: {v}")
            if parts:
                lines.append(f"  - **{year}:** " + " | ".join(parts))

        if not lines:
            return ""

        out = [
            "Income Statement (pre-converted — reproduce Revenue, Gross Profit, Operating Income, Operating Expenses, and Net Income verbatim; read EPS from the filing text):"
        ]
        out.extend(lines)
        return "\n".join(out)

    except Exception as e:
        logger.warning(f"yfinance income statement fetch failed for {ticker}: {e}")
        return ""


def detect_filing_denomination(content: str) -> str:
    """
    Detect whether the SEC filing reports financial figures in thousands or millions.
    Searches the first 200K chars for the standard denomination note used in financial tables.
    Returns an explicit context note to prepend to Financial Performance context.
    """
    sample = content[:200_000] if content else ""
    if re.search(r"\(\s*in\s+thousands\b", sample, re.IGNORECASE) or re.search(
        r"in\s+thousands,?\s+except", sample, re.IGNORECASE
    ):
        return (
            "IMPORTANT — UNITS: This filing reports all financial statement figures IN THOUSANDS. "
            "A raw number like 1,305,375 means $1,305,375 thousand = $1.3B. "
            "Always convert: divide by 1,000 and express in billions (B) or millions (M). "
            "Per-share figures (EPS, dividends) are already in dollars — do not convert them."
        )
    return (
        "IMPORTANT — UNITS: This filing reports all financial statement figures IN MILLIONS. "
        "A raw number like 394,328 means $394,328 million = $394.3B. "
        "Express figures as millions (M) or convert to billions (B). "
        "Per-share figures (EPS, dividends) are already in dollars — do not convert them."
    )


def get_yfinance_cashflow(ticker: str) -> str:
    """
    Fetch operating cash flow and free cash flow from yfinance to supplement
    Financial Performance section when FAISS retrieval misses cash flow statement.
    Returns a formatted string ready to append to the section context.
    """
    try:
        t = yf.Ticker(ticker)
        cf = t.cashflow
        if cf is None or cf.empty:
            return ""

        ocf_keys = [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
            "Cash From Operations",
        ]
        capex_keys = ["Capital Expenditure", "Capital Expenditures", "Purchase Of Ppe"]

        ocf_row = next((cf.loc[k] for k in ocf_keys if k in cf.index), None)
        capex_row = next((cf.loc[k] for k in capex_keys if k in cf.index), None)

        if ocf_row is None:
            return ""

        lines = ["Cash Flow & Capex — reproduce each line below as a bullet exactly as written:"]
        capex_lines = []
        for col in list(cf.columns)[:3]:
            year = col.year
            ocf = int(ocf_row[col]) if ocf_row[col] == ocf_row[col] else None
            if ocf is None:
                continue
            ocf_b = f"${ocf / 1e9:.1f}B"
            if capex_row is not None:
                capex = capex_row[col]
                if capex == capex:
                    capex_val = int(capex)
                    fcf = ocf + capex_val  # capex is negative in yfinance
                    fcf_b = f"${fcf / 1e9:.1f}B"
                    capex_abs = abs(capex_val)
                    capex_fmt = (
                        f"${capex_abs / 1e9:.3f}B"
                        if capex_abs >= 1e9
                        else f"${capex_abs / 1e6:.0f}M"
                    )
                    lines.append(
                        f"  - **{year}:** Operating Cash Flow {ocf_b} | Free Cash Flow {fcf_b}"
                    )
                    capex_lines.append(f"  - **{year}:** Capex {capex_fmt}")
                    continue
            lines.append(f"  - **{year}:** Operating Cash Flow {ocf_b}")

        if capex_lines:
            lines.append("Capital Expenditures on property and equipment — reproduce verbatim:")
            lines.extend(capex_lines)

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        logger.warning(f"yfinance cashflow fetch failed for {ticker}: {e}")
        return ""


# ---------------------------------------------------------------------------
# LLM output post-processing: strip absence sentences
# ---------------------------------------------------------------------------

_ABSENCE_LINE_RE = re.compile(
    r"(?i)\b("
    r"none\s+explicitly\s+named"
    r"|no\s+(specific\s+)?(company|mergers?\s+(or|and)\s+acquisitions?|acquisitions?\s+(or|and)\s+divestitures?|mergers?|acquisitions?|divestitures?|competitors?|strategies?)\s+(were\s+|are\s+|have\s+been\s+)?(disclosed|found|named|mentioned|identified|stated)"
    r"|no\s+(mergers?|acquisitions?|divestitures?)\s+(or\s+(mergers?|acquisitions?|divestitures?)\s+)?(were\s+|are\s+|have\s+been\s+)?(disclosed|found|named|mentioned|identified|stated)"
    r"|no\s+notable\s+subsidiaries?"
    r"|does?\s+not\s+mention\s+any"
    r"|does?\s+not\s+explicitly\s+(state|mention|name|disclose)"
    r"|is\s+not\s+(explicitly\s+)?stated"
    r"|are\s+not\s+explicitly"
    r"|not\s+explicitly\s+stated"
    r"|not\s+mentioned\s+in\s+the"
    r"|priorities\s+are\s+not\s+explicitly"
    r"|strategies?\s+.*not\s+explicitly"
    r"|expansion\s+plans?\s+.*not\s+explicitly"
    r"|total\s+amount\s+.*not\s+explicitly"
    r"|no\s+explicit\s+\w+(\s+\w+)?\s+(share|ranking|data|information|detail)\w*\s+(is|was|are|were)?\s*(provided|disclosed|stated|mentioned|found)"
    r"|did\s+not\s+provide\s+(dividend|share|disclosure)"
    r"|not\s+provided\s+in\s+the\s+(filing|context|provided)"
    r"|has\s+not\s+(disclosed|specified|provided)\s+the\s+(total|dividend|share|amount|per\s+share)"
    r"|have\s+not\s+(disclosed|specified|provided)"
    r"|does\s+not\s+provide\s+specific\s+(share|dividend|repurchase|amount|total)"
    r"|exact\s+amount\s+is\s+not\s+specified"
    r"|is\s+not\s+obligated\s+to"
    r"|not\s+specified\s+in\s+the\s+filing"
    r"|no\s+explicit\s+(market\s+share|ranking|competitor)"
    r"|no\s+specific\s+competitor\s+names?\s+(appear|are|were|found)"
    r")\b",
    re.IGNORECASE,
)

# Lines that belong in wrong sections or should be stripped unconditionally
_WRONG_SECTION_LINE_RE = re.compile(
    r"(?i)("
    r"selling,?\s+general\s+and\s+administrative"  # SG&A leaking into R&D
    r"|variable\s+selling\s+expenses?"  # SG&A detail
    r"|uncertain\s+tax\s+positions?"  # MD&A/tax item in Risk Factors
    r"|critical\s+accounting\s+estimates?"  # MD&A item in Risk Factors
    r"|accounting\s+standard\s+adoption"  # ASU in any section
    # Per-tranche share repurchase lines (date ranges with prices)
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+,\s*\d{4}\s+to\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    # Model-generated YoY progression sentences ("increased from $X ... and further to $Y")
    r"|and\s+further\s+to\s+[\$\d]"
    r"|also\s+(increased|decreased)\s+from\s+\$[\d]"
    # Orphaned "was as follows:" table header (appears when per-tranche data lines are stripped)
    r"|share\s+repurchase\s+activity\s+during"
    r"|open\s+market\s+and\s+privately\s+negotiated\s+purchases"
    # Useless meta-sentences generated by the model to caption its own output
    r"|for\s+each\s+year\s+(is|are)\s+as\s+stated\s+in\s+the\s+(filing|context|supplemental)"
    r"|(as\s+stated|as\s+provided)\s+in\s+the\s+(filing|supplemental\s+context)"
    r"|the\s+company['']?s\s+total\s+net\s+sales\s+(increased|decreased)\s+from\s+20\d\d\s+to\s+20\d\d\s+and\s+then"
    # Generic YoY direction sentences — model generates these but gets direction wrong; strip all forms
    r"|the\s+company['']?s\s+\w[\w\s,]+\s+(increased|decreased)\s+from\s+20\d\d\s+to\s+20\d\d\s+and\s+then\s+to\s+20\d\d"
    r"|the\s+company['']?s\s+\w[\w\s,]+(increased|decreased)\s+from\s+20\d\d\s+to\s+20\d\d\."
    # Model-generated YoY % sentences ("increased/decreased $X or Y% from 20XX to 20XX")
    # Simple pattern: any line containing "increased/decreased $X or Y% from 20XX"
    r"|(increased|decreased)\s+\$[\d.,]+[BM]?\s+or\s+[\d.,]+%\s+from\s+20\d\d"
    # Financial/accounting content leaking into wrong sections
    r"|classified\s+within\s+level\s+[123]"
    r"|fair\s+value\s+hierarch"
    r"|consolidated\s+balance\s+sheets?"
    r"|non-current\s+asset"
    r"|senior\s+notes?\s+(on|are|will)"  # debt legal entity leaking into subsidiaries
    r"|obligor\s+group"
    r"|additional\s+paid.in.capital"  # balance sheet leaking into strategic initiatives
    r"|accumulated\s+deficit"
    r"|as\s+revised"  # restatement language
    r")\b",
    re.IGNORECASE,
)


def _clean_absence_sentences(text: str) -> str:
    """
    Remove lines that report missing information (e.g. 'None explicitly named',
    'No specific acquisitions are named') and any sub-section heading that becomes
    empty after those lines are removed.
    """
    lines = text.split("\n")

    # Pass 1: drop absence lines and wrong-section lines
    kept = [
        ln
        for ln in lines
        if not _ABSENCE_LINE_RE.search(ln) and not _WRONG_SECTION_LINE_RE.search(ln)
    ]

    # Pass 2: drop sub-section headings (### or **Bold**) whose content block is
    # empty or contains only citation-only lines like "*Citation: Item 1 - Business"
    _HEADING_RE = re.compile(r"^(#{1,4}\s|\*{1,2}[A-Z])")
    _CITATION_ONLY_RE = re.compile(r"^\*?(Citation:|Item\s+\d)", re.IGNORECASE)
    result = []
    i = 0
    while i < len(kept):
        ln = kept[i]
        if _HEADING_RE.match(ln.strip()):
            j = i + 1
            has_content = False
            while j < len(kept):
                ahead = kept[j].strip()
                if _HEADING_RE.match(ahead):
                    break
                if ahead and not _CITATION_ONLY_RE.match(ahead):
                    has_content = True
                    break
                j += 1
            if not has_content:
                i = j  # skip heading and its citation-only trailer
                continue
        result.append(ln)
        i += 1

    # Collapse runs of 3+ blank lines down to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(result))
    return cleaned.strip()


# ---------------------------------------------------------------------------
# LLM output validation with retry
# ---------------------------------------------------------------------------


def validate_llm_response(response: str, context: str = "summary") -> str:
    """Validate LLM output quality. Raises ValueError to trigger retry."""
    if not response or len(response.strip()) < 50:
        raise ValueError("LLM returned empty or too-short response")

    refusal_patterns = ["I don't have", "I cannot", "I'm unable", "As an AI"]
    if any(p in response for p in refusal_patterns):
        raise ValueError("LLM refused to answer - retry with simplified prompt")

    if context == "financial":
        financial_signals = ["$", "%", "revenue", "income", "billion", "million"]
        if not any(s in response.lower() for s in financial_signals):
            raise ValueError("No financial signal in response - retry")

    # Detect repetition loop: any 8-word phrase appearing more than 6 times
    # Threshold is 6 (not 3) because structured sections like Risk Factors legitimately
    # repeat short structural phrases (e.g. "Nature of the risk:") across multiple entries.
    words = response.split()
    if len(words) > 60:
        for i in range(len(words) - 8):
            phrase = " ".join(words[i : i + 8])
            if response.count(phrase) > 6:
                raise ValueError("LLM stuck in repetition loop - retry")

    return response.strip()


async def generate_section(llm, messages: list, context: str = "summary") -> str:
    """Generate LLM response with up to 3 retries, exponential backoff + jitter."""
    for attempt in range(3):
        try:
            response = await llm.ainvoke(messages)
            validated = validate_llm_response(response.content, context)
            return _clean_absence_sentences(validated)
        except ValueError as e:
            if attempt < 2:
                wait = (2**attempt) + random.uniform(0, 1)  # 1-2s, 2-3s
                logger.warning(
                    f"Validation failed (attempt {attempt + 1}): {e}. Retrying in {wait:.1f}s..."
                )
                await asyncio.sleep(wait)
            else:
                logger.error(f"All retries failed: {e}")
                return "Analysis could not be generated for this section."
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = (2**attempt) * 15 + random.uniform(0, 5)  # 15-20s, 30-35s, 60-65s
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/3). Waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"LLM error: {e}")
                return "Unable to retrieve information for this section."
    return "Unable to retrieve information for this section."


# ---------------------------------------------------------------------------
# Section summary prompts
# ---------------------------------------------------------------------------

# SECTION_PROMPTS: each entry is (retrieval_query, llm_instruction, context_type, lambda_mult, k)
# retrieval_query  - keyword-focused, used for FAISS semantic search
# llm_instruction  - detailed instruction sent to the LLM with the retrieved context
# context_type     - used for response validation ("summary" or "financial")
# lambda_mult      - MMR diversity: 1.0 = pure relevance, 0.0 = pure diversity
# k                - number of chunks to retrieve
SECTION_PROMPTS = {
    "Business Overview": (
        "business model products services revenue streams customer segments subsidiaries",
        "Analyze Item 1 (Business) of this SEC filing. "
        "Cover: (1) core business model and what the company does, "
        "(2) primary products/services and the most recent year's net sales (one number per line item — do not list prior years here), "
        "(3) key end markets and customer types — describe the company's actual customers; do not list competitors or competitor categories here, "
        "(4) notable operating subsidiaries or reportable business segments — only include if the filing explicitly names them as business operations; skip legal entities created solely for debt issuance (e.g. 'Obligor Group', 'Issuer') and skip entirely if none are named.",
        "summary",
        0.5,  # moderate diversity: products + services + markets + segments
        12,
    ),
    "Financial Performance": (
        "revenue gross profit operating income net income EPS earnings per share cash flow from operations capital expenditures free cash flow",
        "Analyze Item 8 (Financial Statements) and Item 7 (MD&A) of this SEC filing. "
        "Format every metric as: **[year1]:** $X — **[year2]:** $Y — **[year3]:** $Z, using the three distinct fiscal years from the filing. Each year label must be different — never repeat the same year twice. Use the period-end year as stated in the filing. "
        "If the filing states a YoY change verbatim (e.g. 'increased $25,126 million or 6.4%'), add it after the figures — do NOT generate your own 'increased/decreased from' phrasing. "
        "CRITICAL — use ONLY the supplemental figures at the end of this context for ALL dollar amounts. "
        "Do NOT read any revenue, profit, income, or EPS values from the filing text — the supplemental provides pre-converted values for everything. "
        "Cover: (1) total net sales — use supplemental Revenue figures, "
        "(2) gross profit — use supplemental Gross Profit figures (omit gross margin % entirely), "
        "(3) operating income and operating expenses — use supplemental figures, "
        "(4) net income — use supplemental Net Income figure; EPS (basic and diluted) — read from the filing text (EPS is per-share so no denomination conversion needed), "
        "(5) free cash flow and operating cash flow — reproduce the supplemental cash flow bullets word-for-word. "
        "Every dollar figure must come from the supplemental context.",
        "financial",
        0.65,  # high relevance but some diversity to capture both income stmt and cash flow stmt
        15,  # extra chunks: needs income stmt + balance sheet + cash flow statement tables
    ),
    "Risk Factors": (
        "risk factors operational regulatory competitive macro market cybersecurity geopolitical",
        "Analyze Item 1A (Risk Factors) of this SEC filing only — do not include accounting estimates, tax positions, critical accounting policies, or MD&A content. "
        "Select exactly ONE risk for each of these 5 categories (no more than one per category): "
        "(1) Operational — supply chain, manufacturing, or execution risks, "
        "(2) Regulatory/Legal — lawsuits, government investigations, compliance, "
        "(3) Macroeconomic — economic conditions, currency, interest rates, "
        "(4) Competitive — market competition, product transitions, "
        "(5) Technology/Cybersecurity — data, privacy, cybersecurity risks. "
        "For each risk provide: a **bold title**, one sentence on the nature of the risk, one sentence on the potential business impact, and an italic citation. "
        "For risks involving lawsuits cite *Item 3 - Legal Proceedings*.",
        "summary",
        0.5,  # moderate diversity: need different risk categories
        12,
    ),
    "Strategic Initiatives": (
        "research development expense R&D headcount infrastructure acquisitions M&A share repurchase buyback dividends declared capital expenditure growth strategy",
        "Analyze Items 1 and 7 (Business and MD&A) of this SEC filing. "
        "Cover: (1) R&D investments — use only the line item specifically labeled 'Research and development' (a sub-line under operating expenses); do NOT use the total operating expenses figure; state the R&D dollar amount verbatim for each year and include what drove the increase if stated in the filing; do not include SG&A or administrative expense details, "
        "(2) M&A — list only company names acquired or divested with deal value; skip entirely if none named, "
        "(3) capital allocation — share repurchase totals, dividend per share by year, and capex on property/equipment by year; for capex use the supplemental Capital Expenditures figures verbatim; exclude all tax/deferred items; 2-3 bullets, no per-tranche tables, "
        "(4) stated growth strategies — include only if distinct strategies beyond R&D are explicitly stated; skip if only generic technology language is present. "
        "Do not mention ASU updates, FASB pronouncements, or accounting standard adoptions.",
        "summary",
        0.6,  # slightly relevance-heavy: R&D expense chunks must rank over reseller/general chunks
        12,
    ),
    "Market Position": (
        "competitors competitive landscape market share advantages patents moats brand network effects ecosystem",
        "Analyze Item 1 (Business) of this SEC filing. "
        "Cover exactly these 4 topics and no others: "
        "(1) competitive landscape and key dynamics — do NOT include market share data here (that goes in topic 2), "
        "(2) estimated market share or ranking — only if explicitly stated in the filing, "
        "(3) competitive strengths — list at most 5 strengths explicitly stated or clearly implied (brand, ecosystem, switching costs, scale, network effects, patents); present as direct statements; do NOT include negative events, incidents, or lawsuits as strengths, "
        "(4) named competitors — only if specific competitor names appear in context; skip entirely if none. "
        "Do not add sections on human capital, seasonality, ESG, legal proceedings, or any other topic.",
        "summary",
        0.5,  # moderate diversity: competitive landscape + moats
        12,
    ),
}

EXECUTIVE_SUFFIX = (
    "\n\nFormat: exactly 4-5 bullet points, one key insight per bullet. "
    "Bold every metric. No filler sentences. Lead each bullet with the insight, not the section name."
)
ANALYST_SUFFIX = (
    "\n\nFormat: use `###` subheadings for each major topic, "
    "bullet points under each heading, and **bold** for every figure. "
    "Include specific citations in italics (e.g., *Item 7 - MD&A*) and YoY comparisons where available. "
    "Only cite figures that appear verbatim in the filing — do not calculate or derive any number."
)


# ---------------------------------------------------------------------------
# Realtime keywords for query classification
# ---------------------------------------------------------------------------

REALTIME_KEYWORDS = [
    "price",
    "stock price",
    "share price",
    "market cap",
    "p/e",
    "pe ratio",
    "today",
    "current price",
    "right now",
    "52 week",
    "52-week",
    "trading at",
    "eps",
    "earnings per share",
    "dividend",
    "yield",
]

NEWS_KEYWORDS = [
    "news",
    "headline",
    "recent",
    "latest news",
    "announcement",
    "sentiment",
    "press release",
    "analyst rating",
    "upgrade",
    "downgrade",
    "article",
]

# Casual / small-talk patterns - politely redirect to financial topics
CASUAL_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|howdy|greetings|sup|yo|hiya)\W*$", re.IGNORECASE),
    re.compile(
        r"^\s*(how are you|how do you do|how\'?s it going|what\'?s up|how r u)\W*$", re.IGNORECASE
    ),
    re.compile(
        r"^\s*(thanks?|thank you|ty|thx|cheers|great|awesome|cool|nice|ok|okay)\W*$", re.IGNORECASE
    ),
    re.compile(
        r"^\s*(who are you|what are you|what\'?s your name|introduce yourself)\W*$", re.IGNORECASE
    ),
    re.compile(r"^\s*(good (morning|afternoon|evening|night))\W*$", re.IGNORECASE),
    re.compile(r"^\s*(bye|goodbye|see you|cya|later)\W*$", re.IGNORECASE),
]


def is_casual_chat(question: str) -> bool:
    return any(p.match(question.strip()) for p in CASUAL_PATTERNS)


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------


async def process_filing_background(
    ticker: str, job_id: str, analyze_mode: str = "analyst", form_type: str = "10-K"
) -> None:
    """
    Full filing processing pipeline: download → index → summarize → charts.
    Supports form_type="10-K" (annual) or "10-Q" (quarterly).
    Updates jobs[job_id] with progress throughout.
    """
    ticker = ticker.upper()

    # Per-ticker lock prevents race conditions on concurrent requests for same ticker
    if ticker not in ticker_locks:
        ticker_locks[ticker] = asyncio.Lock()

    async with ticker_locks[ticker]:
        try:

            def update(progress: str, status: str = "processing") -> None:
                jobs[job_id] = {**jobs[job_id], "progress": progress, "status": status}
                logger.info(f"[{ticker}] {progress}")

            update("Fetching filing from SEC EDGAR...")

            set_identity(os.getenv("SEC_IDENTITY", "FinSight AI finsight@example.com"))
            company = Company(ticker)

            company_profile = {
                "ticker": ticker,
                "name": company.name,
                "cik": company.cik,
                "tickers": company.tickers,
                "sic_code": company.sic,
                "industry": company.industry,
                "fiscal_year_end": company.fiscal_year_end,
                "exchanges": company.get_exchanges(),
            }
            try:
                addr = company.business_address()
                if addr:
                    company_profile["business_address"] = {
                        "street": addr.street1,
                        "city": addr.city,
                        "state": addr.state_or_country,
                        "zip": addr.zipcode,
                    }
            except Exception:
                pass

            filings = company.get_filings(form=form_type)
            if not filings:
                raise ValueError(f"No {form_type} filings found for {ticker}")

            latest_filing = filings[0]
            accession_number = latest_filing.accession_number
            filing_date = str(latest_filing.filing_date)

            # Check cache BEFORE downloading text - text download is the slow step
            if faiss_manager.needs_rebuild(ticker, accession_number, form_type):
                # New or changed filing - evict stale analysis cache entries
                stale_keys = [
                    k for k in analysis_cache if k == ticker or k.startswith(f"{ticker}:")
                ]
                for k in stale_keys:
                    del analysis_cache[k]
                if stale_keys:
                    logger.info(f"[{ticker}] Evicted stale analysis cache (new filing detected)")

                update("Extracting text from filing...")
                content = await asyncio.to_thread(latest_filing.text)
                update("Building vector index...")
                vector_store = faiss_manager.create_vector_store(
                    content=content,
                    ticker=ticker,
                    accession_number=accession_number,
                    form_type=form_type,
                )
                if not vector_store:
                    raise ValueError("Failed to build vector index")

                # Upload new FAISS store to HF Dataset repo in background (non-blocking)
                if hf_store.is_configured():
                    store_path = faiss_manager._get_store_path(ticker, form_type)
                    asyncio.create_task(
                        asyncio.to_thread(hf_store.upload_store, store_path, ticker, form_type)
                    )
            else:
                update("Loading cached vector index...")
                content = None  # not needed - FAISS store is current
                vector_store = faiss_manager.load_store(ticker, form_type)
                if not vector_store:
                    raise ValueError("Cached store missing - please re-run analysis")

            # Generate 5-section summary
            llm = get_llm()
            suffix = EXECUTIVE_SUFFIX if analyze_mode == "executive" else ANALYST_SUFFIX
            sections = []

            # Parallelise FAISS retrieval for all 5 sections (local, no API cost)
            # Each section gets its own retriever tuned with a section-specific lambda_mult.
            section_items = list(SECTION_PROMPTS.items())

            if faiss_manager.load_store(ticker, form_type):
                retrievers = [
                    faiss_manager.get_mmr_retriever(
                        ticker, form_type=form_type, k=k, lambda_mult=lm
                    )
                    for _, (_, _, _, lm, k) in section_items
                ]
                all_docs = await asyncio.gather(
                    *[
                        asyncio.to_thread(r.invoke, rq)
                        for r, (_, (rq, _, _, _, _)) in zip(retrievers, section_items)
                    ]
                )
                contexts = ["\n\n---\n\n".join(d.page_content for d in docs) for docs in all_docs]

                # Detect denomination (thousands vs millions) from filing text and prepend to
                # Financial Performance AND Business Overview contexts.
                denom_note = detect_filing_denomination(content or "")
                fin_idx = list(SECTION_PROMPTS.keys()).index("Financial Performance")
                biz_idx = list(SECTION_PROMPTS.keys()).index("Business Overview")
                strat2_idx = list(SECTION_PROMPTS.keys()).index("Strategic Initiatives")
                contexts[fin_idx] = denom_note + "\n\n" + contexts[fin_idx]
                contexts[biz_idx] = denom_note + "\n\n" + contexts[biz_idx]
                contexts[strat2_idx] = denom_note + "\n\n" + contexts[strat2_idx]

                # Augment Financial Performance with yfinance income statement (pre-converted units)
                # This completely bypasses LLM denomination arithmetic for revenue/profit/income
                inc_data = await asyncio.to_thread(get_yfinance_income_stmt, ticker)
                if inc_data:
                    contexts[fin_idx] += f"\n\n---\n\n{inc_data}"
                    # Also inject most-recent-year revenue into Business Overview to prevent
                    # FAISS from substituting cost-of-revenue chunks for the net sales line
                    rev_note = next(
                        (ln for ln in inc_data.splitlines() if ln.strip().startswith("- **")),
                        None,
                    )
                    if rev_note:
                        contexts[biz_idx] += (
                            f"\n\n---\n\nRevenue reference (most recent year — use for net sales line): {rev_note.strip()}"
                        )

                # Augment Financial Performance with yfinance cash flow (FAISS rarely retrieves it)
                cf_data = await asyncio.to_thread(get_yfinance_cashflow, ticker)
                if cf_data:
                    contexts[fin_idx] += f"\n\n---\n\n{cf_data}"

                # Augment Strategic Initiatives with yfinance dividends + capex
                strat_idx = list(SECTION_PROMPTS.keys()).index("Strategic Initiatives")
                div_data = await asyncio.to_thread(get_yfinance_dividends, ticker)
                if div_data:
                    contexts[strat_idx] += f"\n\n---\n\n{div_data}"
                else:
                    # Explicit signal so model doesn't hallucinate dividend figures
                    contexts[strat_idx] += (
                        "\n\nDividend note: No dividend data found for this company — do not include dividend per share in Capital Allocation."
                    )
                # Inject cash flow supplement into Strategic Initiatives so capex figures are available
                if cf_data:
                    contexts[strat_idx] += f"\n\n---\n\n{cf_data}"
                # Augment Strategic Initiatives with yfinance R&D (FAISS retrieves total opex instead of R&D sub-line)
                rd_data = await asyncio.to_thread(get_yfinance_rd, ticker)
                if rd_data:
                    contexts[strat_idx] += f"\n\n---\n\n{rd_data}"
                # Note: SEC share repurchase tables report share counts in thousands
                contexts[strat_idx] += (
                    "\n\nNote: Share counts in SEC filings are in thousands (e.g., '89,498' means 89,498 thousand = 89.5 million shares)."
                )
            elif content:
                contexts = [content[:3000]] * len(section_items)
            else:
                contexts = ["Filing text not available."] * len(section_items)

            # LLM calls remain sequential to respect NVIDIA NIM rate limits
            for i, ((section_title, (_, llm_instruction, context_type, _lm, _k)), ctx) in enumerate(
                zip(section_items, contexts), 1
            ):
                update(f"Generating summary ({i}/5): {section_title}...")

                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Company: {company.name} ({ticker})\n"
                        f"Context from {form_type} filing:\n{ctx}\n\n"
                        f"Task: {llm_instruction}{suffix}\n\n"
                        f"Write naturally: introduce the company as '{company.name}' on first reference, "
                        f"then vary with 'the company', pronouns, or short forms as a financial analyst would."
                    ),
                ]

                _t = time.time()
                answer = await generate_section(llm, messages, context=context_type)
                logger.info(
                    f"[{ticker}] Section {i}/5 ({section_title}) took {time.time() - _t:.1f}s | {len(answer)} chars"
                )
                sections.append({"title": section_title, "content": answer})

            update("Generating financial charts...")
            try:
                chart_company, chart_xbrs = await asyncio.to_thread(
                    get_company_filings_data, ticker
                )
                income_plot = await asyncio.to_thread(
                    plot_revenue, ticker, chart_company, chart_xbrs
                )
                balance_plot = await asyncio.to_thread(
                    plot_balance_sheet, ticker, chart_company, chart_xbrs
                )
                cashflow_plot = await asyncio.to_thread(
                    plot_cash_flow, ticker, chart_company, chart_xbrs
                )
                financial_charts = {
                    "income_statement": income_plot,
                    "balance_sheet": balance_plot,
                    "cash_flow": cashflow_plot,
                }
            except Exception as chart_err:
                logger.warning(f"Chart generation failed for {ticker}: {chart_err}")
                financial_charts = {}

            result = {
                "ticker": ticker,
                "company_name": company.name,
                "filing_date": filing_date,
                "company_profile": company_profile,
                "summary": sections,
                "financial_charts": financial_charts,
                "analyze_mode": analyze_mode,
                "form_type": form_type,
                "provider": get_provider_name(),
            }

            # Only cache if all sections generated successfully
            all_ok = all(
                "Unable to retrieve information" not in str(s.get("content", "")) for s in sections
            )
            if all_ok:
                analysis_cache[f"{ticker}:{form_type}:{analyze_mode}"] = result
                analysis_cache[ticker] = result  # alias: always store latest by bare ticker too
            jobs[job_id] = {
                "job_id": job_id,
                "status": "complete",
                "progress": "Complete",
                "result": result,
            }
            logger.info(f"[{ticker}] Processing complete")

        except Exception as e:
            logger.error(f"[{ticker}] Processing failed: {e}")
            jobs[job_id] = {
                "job_id": job_id,
                "status": "error",
                "progress": f"Failed: {str(e)}",
                "error": str(e),
            }
        finally:
            active_jobs.discard(job_id)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _cleanup_jobs():
    """Remove completed/error jobs older than 30 minutes to prevent unbounded memory growth."""
    while True:
        await asyncio.sleep(300)  # run every 5 minutes
        cutoff = time.time() - 1800  # 30 minutes
        stale = [
            jid
            for jid, job in jobs.items()
            if job.get("status") in ("complete", "error")
            and job.get("created_at", time.time()) < cutoff
        ]
        for jid in stale:
            jobs.pop(jid, None)
        if stale:
            logger.info(f"Cleaned up {len(stale)} stale jobs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinSight AI starting up")
    asyncio.create_task(_cleanup_jobs())
    # Restore FAISS indexes from HF Dataset repo before accepting requests.
    # No-ops silently if HF_TOKEN / HF_DATASET_REPO are not set (local dev).
    if hf_store.is_configured():
        hf_store.ensure_repo_exists()
        restored = await asyncio.to_thread(hf_store.restore_all_stores, faiss_manager.base_dir)
        logger.info(f"Startup: {restored} FAISS store(s) ready from HF Dataset repo")
    else:
        logger.info("Startup: HF not configured - FAISS will build on first request per ticker")
    # Pre-load BGE embeddings model so the first user request has no extra latency.
    await asyncio.to_thread(faiss_manager.get_embeddings_model)
    logger.info("Startup: BGE embeddings model ready")
    yield
    logger.info("FinSight AI shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinSight AI",
    description="SEC 10-K/10-Q Analysis - NVIDIA NIM / Ollama · LangChain · FAISS · BGE-base",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Supports multiple origins: set FRONTEND_URL=http://localhost:3001,https://name.vercel.app
    allow_origins=os.getenv("FRONTEND_URL", "http://localhost:3001").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ---------------------------------------------------------------------------
# Company search helpers
# ---------------------------------------------------------------------------

_company_db: list[dict] = []
_company_db_fetched_at: float = 0.0
_COMPANY_DB_TTL = 86400  # refresh once per day


def fetch_company_database() -> list[dict]:
    """Fetch SEC company list with 24-hour in-memory cache to avoid hammering SEC on every search."""
    global _company_db, _company_db_fetched_at
    if _company_db and (time.time() - _company_db_fetched_at) < _COMPANY_DB_TTL:
        return _company_db

    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    try:
        sec_identity = os.getenv("SEC_IDENTITY", "FinSight AI finsight@example.com")
        response = requests.get(url, headers={"User-Agent": sec_identity}, timeout=10)
        response.raise_for_status()
        data = response.json()
        fields = data["fields"]
        cik_idx = fields.index("cik")
        name_idx = fields.index("name")
        ticker_idx = fields.index("ticker")
        _company_db = [
            {
                "ticker": row[ticker_idx].upper(),
                "cik": str(row[cik_idx]).zfill(10),
                "title": row[name_idx],
            }
            for row in data["data"]
            if row[ticker_idx]
        ]
        _company_db_fetched_at = time.time()
        logger.info(f"Company database fetched: {len(_company_db):,} companies")
        return _company_db
    except Exception as e:
        logger.error(f"Error fetching company database: {e}")
        return _company_db  # return stale cache on failure rather than empty list


def rank_search_results(companies: list[dict], query: str) -> list[dict]:
    def score(c):
        t, n, q = c["ticker"].upper(), c["title"].upper(), query.upper()
        if t == q:
            return 0
        if t.startswith(q):
            return 1
        if n.startswith(q):
            return 2
        return 3

    return sorted(companies, key=score)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "provider": get_provider_name()}


@app.get("/company_search/")
def company_search(q: str = Query(...)):
    if not q or len(q) < 2:
        return {"results": []}
    companies = fetch_company_database()
    q_upper = q.upper()
    matching = [
        c for c in companies if q_upper in c["title"].upper() or q_upper in c["ticker"].upper()
    ]
    return {"results": rank_search_results(matching, q)[:10]}


@app.post("/analyze_filing/")
async def analyze_filing(
    ticker: str,
    bg: BackgroundTasks,
    analyze_mode: str = Query("analyst", description="'executive' or 'analyst'"),
    form_type: str = Query(
        "10-K", description="Filing type: '10-K' (annual) or '10-Q' (quarterly)"
    ),
):
    """
    Initiate async filing download and analysis.
    Returns job_id immediately - poll /job_status/{job_id} for progress.
    """
    ticker = ticker.upper()
    cache_key = f"{ticker}:{form_type}:{analyze_mode}"

    # Return cached result instantly
    if cache_key in analysis_cache:
        job_id = f"cached_{ticker}_{uuid4().hex[:8]}"
        jobs[job_id] = {
            "job_id": job_id,
            "status": "complete",
            "progress": "Loaded from cache",
            "result": analysis_cache[cache_key],
        }
        return {"job_id": job_id}

    job_id = str(uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": "Starting...",
        "created_at": time.time(),
    }
    active_jobs.add(job_id)
    bg.add_task(process_filing_background, ticker, job_id, analyze_mode, form_type)
    return {"job_id": job_id}


@app.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "processing"),
        progress=job.get("progress", ""),
        result=job.get("result"),
        error=job.get("error"),
    )


@app.delete("/cache/{ticker}")
async def clear_cache(ticker: str):
    """Remove all variants for a ticker from the analysis cache."""
    ticker = ticker.upper()
    removed = [k for k in list(analysis_cache.keys()) if k == ticker or k.startswith(f"{ticker}:")]
    for k in removed:
        del analysis_cache[k]
    return {"cleared": removed}


@app.delete("/cache/{ticker}/{form_type}/{mode}")
async def clear_cache_variant(ticker: str, form_type: str, mode: str):
    """Remove a specific ticker/form_type/mode variant from the analysis cache."""
    ticker = ticker.upper()
    form_type = form_type.upper().replace("10K", "10-K").replace("10Q", "10-Q")
    mode = mode.lower()
    key = f"{ticker}:{form_type}:{mode}"
    removed = []
    if key in analysis_cache:
        del analysis_cache[key]
        removed.append(key)
    # Also clear bare-ticker alias if it pointed to this variant
    if ticker in analysis_cache:
        del analysis_cache[ticker]
        removed.append(ticker)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No cache entry found for {ticker} · {form_type} · {mode.capitalize()}",
        )
    return {"cleared": f"{ticker} · {form_type} · {mode.capitalize()}"}


@app.get("/ask/")
async def ask(
    question: str,
    ticker: str,
    session_id: str,
    form_type: str = Query("10-K"),
):
    """
    SSE streaming Q&A endpoint.

    Pipeline:
    1. classify_query - keyword-based routing: "document" vs "realtime"
    2. retrieve_documents - FAISS MMR with ticker metadata filter
       OR fetch_realtime_data - yfinance live market data
    3. generate_response - stream tokens via LLM, include conversation history
    4. Final event: { done: true, source, cited_sections }
    """
    ticker = ticker.upper()

    async def token_stream():
        cited_sections: list[str] = []
        source = "document"

        try:
            # ── Casual chat guardrail ────────────────────────────────────────
            if is_casual_chat(question):
                cached = analysis_cache.get(ticker) or analysis_cache.get(
                    f"{ticker}:{form_type}:analyst"
                )
                company_name = cached.get("company_name", ticker) if cached else ticker
                greeting = (
                    f"Hello! I'm **FinSight AI** - a financial analyst specialized in SEC 10-K and 10-Q filings.\n\n"
                    f"I can help you analyze **{company_name} ({ticker})**. Try asking about:\n"
                    f"- Revenue growth and profit margins\n"
                    f"- Key risk factors\n"
                    f"- Strategic initiatives and R&D investments\n"
                    f"- Competitive landscape\n\n"
                    f"What would you like to know?"
                )
                yield f"data: {json.dumps({'token': greeting})}\n\n"
                yield f"data: {json.dumps({'done': True, 'source': 'document', 'cited_sections': []})}\n\n"
                return

            # ── Step 1: Classify ────────────────────────────────────────────
            q_lower = question.lower()
            is_realtime = any(kw in q_lower for kw in REALTIME_KEYWORDS)
            is_news = any(kw in q_lower for kw in NEWS_KEYWORDS)

            # ── Step 2a: Fetch news headlines ────────────────────────────────
            news_context = None
            if is_news:
                try:
                    import feedparser

                    rss_url = (
                        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                        f"?s={ticker}&region=US&lang=en-US"
                    )
                    feed = await asyncio.to_thread(feedparser.parse, rss_url)
                    headlines = [e.title for e in feed.entries[:8]]
                    if headlines:
                        news_context = f"**Recent news headlines for {ticker}:**\n" + "\n".join(
                            f"- {h}" for h in headlines
                        )
                        source = "realtime"
                except Exception as news_err:
                    logger.warning(f"News RSS fetch failed for {ticker}: {news_err}")
                    news_context = None

            # ── Step 2b: Fetch realtime market data ─────────────────────────
            live_context = None
            if is_realtime:
                try:
                    # yfinance is synchronous - run in thread to avoid blocking event loop
                    info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                    price = info.get("currentPrice") or info.get("regularMarketPrice")
                    mktcap = info.get("marketCap")
                    mktcap_str = f"- Market Cap: ${mktcap:,}\n" if mktcap else ""
                    live_context = (
                        f"**Real-time market data for {ticker}** (source: Yahoo Finance)\n"
                        f"- Current Price: {info.get('currency', 'USD')} {price}\n"
                        f"{mktcap_str}"
                        f"- P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
                        f"- 52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
                        f"- 52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}\n"
                        f"- EPS (TTM): {info.get('trailingEps', 'N/A')}"
                    )
                    source = "realtime"
                except Exception as yf_err:
                    logger.warning(f"yfinance failed for {ticker}: {yf_err}")
                    live_context = None  # fall through to document retrieval

            # ── Step 2c: Retrieve from FAISS ────────────────────────────────
            doc_context = None
            if not is_realtime or live_context is None:
                retriever = faiss_manager.get_mmr_retriever(ticker, form_type=form_type)
                if retriever:
                    # FAISS retrieval is synchronous - run in thread to avoid blocking event loop
                    docs = await asyncio.to_thread(retriever.invoke, question)
                    doc_context = "\n\n---\n\n".join(d.page_content for d in docs)
                    cited_sections = sorted(
                        set(
                            d.metadata.get("section", "")
                            for d in docs
                            if d.metadata.get("section") and d.metadata["section"] != "General"
                        )
                    )
                    source = "realtime" if live_context else "document"
                    if live_context:
                        source = "hybrid"
                else:
                    doc_context = (
                        f"No filing has been analyzed for {ticker}. "
                        "Please run the analysis first using the Analyze button."
                    )
                    source = "document"

            # ── Build context ────────────────────────────────────────────────
            context_parts = []
            if news_context:
                context_parts.append(news_context)
            if live_context:
                context_parts.append(live_context)
            if doc_context:
                context_parts.append(f"**From {form_type} filing:**\n{doc_context}")
            context = "\n\n".join(context_parts) if context_parts else "No data available."

            # ── Step 3: Conversation history ─────────────────────────────────
            history = session_history.get(session_id, [])
            history_messages = []
            for turn in history[-8:]:  # last 4 turns (8 messages)
                if turn["role"] == "user":
                    history_messages.append(HumanMessage(content=turn["content"]))
                else:
                    history_messages.append(AIMessage(content=turn["content"]))

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                *history_messages,
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
            ]

            # ── Step 4: Stream response ──────────────────────────────────────
            llm = get_llm()
            full_response = ""

            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"

            # Save to session history (cap at SESSION_CAP sessions)
            if session_id not in session_history:
                if len(session_history) >= SESSION_CAP:
                    oldest = next(iter(session_history))
                    del session_history[oldest]
                session_history[session_id] = []

            session_history[session_id].append({"role": "user", "content": question})
            session_history[session_id].append({"role": "assistant", "content": full_response})

            # Final metadata event
            yield f"data: {json.dumps({'done': True, 'source': source, 'cited_sections': cited_sections})}\n\n"

        except Exception as e:
            logger.error(f"SSE error for {ticker}: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# PDF rendering helpers (shared by both export endpoints)
# ---------------------------------------------------------------------------


def _pdf_s(text: str) -> str:
    """Sanitize string to Latin-1 for FPDF (replaces non-encodable chars)."""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


def _pdf_clean_inline(s: str) -> str:
    """Strip markdown bold/italic/code markers, keeping content (Latin-1 safe)."""
    s = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", s)
    s = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", s, flags=re.DOTALL)
    return _pdf_s(s).strip()


def _pdf_render_content(pdf_obj, text: str) -> None:
    """
    Line-by-line markdown → FPDF renderer.
    Handles: ### headings, - bullets, plain paragraphs.
    """
    usable_w = pdf_obj.w - pdf_obj.l_margin - pdf_obj.r_margin
    para_lines: list = []

    def flush_para() -> None:
        if para_lines:
            para = " ".join(para_lines).strip()
            if para:
                pdf_obj.set_font("Helvetica", "", 10)
                pdf_obj.set_text_color(50, 50, 50)
                pdf_obj.set_x(pdf_obj.l_margin)
                pdf_obj.multi_cell(0, 6, para)
                pdf_obj.ln(1)
            para_lines.clear()

    for raw_line in text.split("\n"):
        s = raw_line.strip()
        if re.match(r"^[-*_]{3,}$", s):
            flush_para()
            continue
        if not s:
            flush_para()
            continue
        if s.startswith("#"):
            flush_para()
            heading = _pdf_clean_inline(re.sub(r"^#+\s*", "", s))
            if heading:
                pdf_obj.ln(2)
                pdf_obj.set_font("Helvetica", "B", 10)
                pdf_obj.set_text_color(35, 35, 35)
                pdf_obj.cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
                y_pos = pdf_obj.get_y()
                pdf_obj.set_draw_color(200, 200, 200)
                pdf_obj.set_line_width(0.2)
                pdf_obj.line(
                    pdf_obj.l_margin,
                    y_pos,
                    pdf_obj.l_margin + min(len(heading) * 2.6, usable_w),
                    y_pos,
                )
                pdf_obj.ln(2)
            continue
        if re.match(r"^[-*+\xb7]\s+", s):
            flush_para()
            content = _pdf_clean_inline(re.sub(r"^[-*+\xb7]\s+", "", s))
            if content:
                pdf_obj.set_font("Helvetica", "", 10)
                pdf_obj.set_text_color(50, 50, 50)
                pdf_obj.set_x(pdf_obj.l_margin + 5)
                pdf_obj.multi_cell(usable_w - 5, 6, f"\xb7  {content}")
            continue
        para_lines.append(_pdf_clean_inline(s))

    flush_para()


@app.post("/export_pdf/")
async def export_pdf(payload: dict[str, Any] = Body(...)):
    """
    Generate and download a PDF report from the result payload sent by the frontend.
    No server-side cache dependency - data comes directly from the request body.
    """
    from fpdf import FPDF

    cached = payload
    ticker = cached.get("ticker", "").upper()
    analyze_mode = cached.get("analyze_mode", "analyst")
    form_type = cached.get("form_type", "10-K")
    if not ticker or not cached.get("summary"):
        raise HTTPException(status_code=400, detail="Invalid payload: missing ticker or summary.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 12, _pdf_s(f"{cached['company_name']} ({ticker})"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110, 110, 110)
    filing_form = cached.get("form_type", form_type)
    pdf.cell(
        0,
        5,
        _pdf_s(
            f"SEC {filing_form} Filing Analysis  |  Filing Date: {cached['filing_date']}  |  FinSight AI"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    cached_mode = cached.get("analyze_mode", analyze_mode)
    pdf.cell(
        0,
        5,
        _pdf_s(f"LLM Provider: {cached['provider']}  |  Analysis Mode: {cached_mode.title()}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    # Accent rule
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    # ── Company profile row ───────────────────────────────────────────────────
    profile = cached.get("company_profile", {})
    if profile:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, "Company Profile", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(70, 70, 70)
        metrics_line = "  |  ".join(
            filter(
                None,
                [
                    f"CIK: {profile.get('cik', 'N/A')}",
                    f"SIC: {profile.get('sic_code', 'N/A')}",
                    f"Industry: {profile.get('industry', 'N/A')}",
                    f"Fiscal Year End: {profile.get('fiscal_year_end', 'N/A')}",
                ],
            )
        )
        pdf.cell(0, 5, _pdf_s(metrics_line), new_x="LMARGIN", new_y="NEXT")
        addr = profile.get("business_address")
        if addr:
            parts = [addr.get("street"), addr.get("city"), addr.get("state"), addr.get("zip")]
            address_line = ", ".join(p for p in parts if p)
            if address_line:
                pdf.cell(0, 5, _pdf_s(address_line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # ── Summary sections ──────────────────────────────────────────────────────
    for section in cached.get("summary", []):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, _pdf_s(section["title"]), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(59, 130, 246)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)
        _pdf_render_content(pdf, section.get("content", ""))
        pdf.ln(3)

    # ── News Sentiment ────────────────────────────────────────────────────────
    sentiment_data = cached.get("news_sentiment")
    if sentiment_data and sentiment_data.get("sentiment") != "unavailable":
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 9, "News Sentiment", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(59, 130, 246)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)

        label = sentiment_data.get("sentiment", "neutral").title()
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 6, f"Overall Sentiment: {label}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        _pdf_render_content(pdf, sentiment_data.get("summary", ""))

        headlines = sentiment_data.get("headlines", [])
        if headlines:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 6, "Recent Headlines", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            for h in headlines[:8]:
                title_text = _pdf_clean_inline(h.get("title", ""))
                pub = h.get("published", "")[:16]
                link = h.get("link", "")
                label = f"\xb7  {title_text}  ({pub})" if pub else f"\xb7  {title_text}"
                pdf.set_font("Helvetica", "", 9)
                pdf.set_x(pdf.l_margin + 3)
                if link:
                    pdf.set_text_color(59, 130, 246)
                    pdf.cell(0, 5, label, new_x="LMARGIN", new_y="NEXT", link=link)
                else:
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, label)

    # ── Financial Charts ──────────────────────────────────────────────────────
    charts = cached.get("financial_charts", {})
    chart_labels = {
        "income_statement": "Income Statement",
        "balance_sheet": "Balance Sheet",
        "cash_flow": "Cash Flow Statement",
    }
    if charts:
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        for key, label in chart_labels.items():
            b64 = charts.get(key)
            if not b64:
                continue
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(0, 9, label, new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(59, 130, 246)
            pdf.set_line_width(0.4)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
            try:
                img_bytes = base64.b64decode(b64)
                pdf.image(BytesIO(img_bytes), w=usable_w)
            except Exception as img_err:
                logger.warning(f"Could not embed chart {key}: {img_err}")

    # ── Footer (must be last - disables auto_page_break) ──────────────────────
    pdf.set_auto_page_break(False)
    pdf.set_y(-18)
    pdf.set_line_width(0.3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(160, 160, 160)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(
        0,
        5,
        f"FinSight AI  |  Generated {ts}  |  For informational purposes only. Not financial advice.",
        align="C",
    )

    pdf_bytes = bytes(pdf.output())
    safe_form = filing_form.replace("-", "")  # "10K" or "10Q"
    mode_cap = cached_mode.title()  # "Analyst" or "Executive"
    filename = f"FinSight_AI_{ticker}_{safe_form}_{mode_cap}_{cached['filing_date']}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/export_pdf/compare/")
async def export_pdf_compare(payload: dict[str, Any] = Body(...)):
    """
    Generate a PDF for a side-by-side company comparison.
    Accepts the comparisonData object from the frontend directly.
    """
    from fpdf import FPDF

    t1 = payload.get("ticker1", "").upper()
    t2 = payload.get("ticker2", "").upper()
    c1 = payload.get("company1", {})
    c2 = payload.get("company2", {})
    key_differences = payload.get("key_differences", "")
    analyze_mode = payload.get("analyze_mode", "analyst")

    if not t1 or not t2 or not c1 or not c2:
        raise HTTPException(status_code=400, detail="Invalid comparison payload.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(
        0,
        11,
        _pdf_s(f"{c1.get('name', t1)} vs {c2.get('name', t2)}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(
        0,
        5,
        _pdf_s(
            f"{t1}: {c1.get('form_type','10-K')} ({c1.get('filing_date','')})  |  {t2}: {c2.get('form_type','10-K')} ({c2.get('filing_date','')})"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        5,
        _pdf_s(f"Analysis Mode: {analyze_mode.title()}  |  FinSight AI"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    # ── Company profiles ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(70, 70, 70)
    for ticker_label, company_data in [(t1, c1), (t2, c2)]:
        cp = company_data.get("company_profile", {})
        if not cp:
            continue
        meta = "  |  ".join(
            filter(
                None,
                [
                    f"CIK: {cp.get('cik', 'N/A')}",
                    f"SIC: {cp.get('sic_code', 'N/A')}",
                    f"Industry: {cp.get('industry', 'N/A')}",
                    f"FYE: {cp.get('fiscal_year_end', 'N/A')}",
                ],
            )
        )
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, _pdf_s(f"{ticker_label}:"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _pdf_s(meta), new_x="LMARGIN", new_y="NEXT")
        addr = cp.get("business_address")
        if addr:
            parts = [addr.get("street"), addr.get("city"), addr.get("state"), addr.get("zip")]
            address_line = ", ".join(p for p in parts if p)
            if address_line:
                pdf.cell(0, 5, _pdf_s(address_line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.ln(3)

    # ── Key Differences (LLM analysis) ───────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, "Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)
    _pdf_render_content(pdf, key_differences)
    pdf.ln(4)

    # ── Per-section side-by-side ──────────────────────────────────────────────
    sections1 = {s["title"]: s["content"] for s in c1.get("summary", [])}
    sections2 = {s["title"]: s["content"] for s in c2.get("summary", [])}
    all_titles = list(sections1.keys()) or list(sections2.keys())

    for title in all_titles:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, _pdf_s(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(59, 130, 246)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)

        for ticker_label, content in [
            (_pdf_s(f"{t1} - {c1.get('name','')}"), sections1.get(title, "")),
            (_pdf_s(f"{t2} - {c2.get('name','')}"), sections2.get(title, "")),
        ]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(59, 130, 246)
            pdf.cell(0, 6, ticker_label, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            _pdf_render_content(pdf, content)
            pdf.ln(3)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_auto_page_break(False)
    pdf.set_y(-18)
    pdf.set_line_width(0.3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(160, 160, 160)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(
        0,
        5,
        f"FinSight AI  |  Generated {ts}  |  For informational purposes only. Not financial advice.",
        align="C",
    )

    pdf_bytes = bytes(pdf.output())
    ft = c1.get("form_type", "10-K").replace("-", "")
    mode_cap = analyze_mode.title()
    filename = f"FinSight_AI_{t1}_vs_{t2}_{ft}_{mode_cap}_{datetime.now().strftime('%Y-%m-%d')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/compare/")
async def compare(ticker1: str = Query(...), ticker2: str = Query(...)):
    """
    Side-by-side comparison of two analyzed companies.
    Both tickers must already be in analysis_cache (run /analyze_filing/ first).
    Returns each company's summary sections plus an LLM-generated key differences analysis.
    """
    t1, t2 = ticker1.upper(), ticker2.upper()
    c1 = analysis_cache.get(t1)
    c2 = analysis_cache.get(t2)
    missing = [t for t, c in [(t1, c1), (t2, c2)] if not c]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Please re-analyze {', '.join(missing)} before comparing.",
        )

    def _section_text(cached: dict, title: str) -> str:
        for s in cached.get("summary", []):
            if s["title"] == title:
                return s["content"]
        return ""

    # Build comparison context (truncated per section to keep prompt manageable)
    comparison_context = ""
    for title in SECTION_PROMPTS:
        s1 = _section_text(c1, title)[:1200]
        s2 = _section_text(c2, title)[:1200]
        comparison_context += f"\n\n**{title}**\n{t1}: {s1}\n{t2}: {s2}"

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"You are comparing {c1['company_name']} ({t1}) vs {c2['company_name']} ({t2}) "
                f"based on their most recent SEC filings.\n\n"
                f"You MUST always generate a full comparison - never say 'unable to retrieve'. "
                f"Use only the filing data provided below. If a specific metric is not available, "
                f"note it briefly and move on.\n\n"
                f"Structure your response with these exact sections:\n"
                f"### Key Differences\n"
                f"List 5 specific differences (business model, revenue scale, margins, risk profile, strategy).\n"
                f"### Notable Similarities\n"
                f"List 2-3 genuine similarities.\n"
                f"### Investment Considerations\n"
                f"1-2 sentences on what each company's filing suggests about its near-term outlook.\n\n"
                f"Use **bold** for all metrics and company names. Be specific with numbers where available.\n\n"
                f"Filing data:{comparison_context}"
            )
        ),
    ]
    key_differences = await generate_section(llm, messages, context="summary")

    return {
        "ticker1": t1,
        "ticker2": t2,
        "company1": {
            "name": c1["company_name"],
            "filing_date": c1["filing_date"],
            "form_type": c1.get("form_type", "10-K"),
            "summary": c1["summary"],
        },
        "company2": {
            "name": c2["company_name"],
            "filing_date": c2["filing_date"],
            "form_type": c2.get("form_type", "10-K"),
            "summary": c2["summary"],
        },
        "key_differences": key_differences,
    }


@app.get("/news_sentiment/")
async def news_sentiment(ticker: str = Query(...)):
    """
    Fetch latest news headlines for a ticker via Yahoo Finance RSS
    and return an LLM-generated sentiment summary.
    """
    import feedparser

    ticker = ticker.upper()
    rss_url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline" f"?s={ticker}&region=US&lang=en-US"
    )

    try:
        feed = await asyncio.to_thread(feedparser.parse, rss_url)
        headlines = [
            {
                "title": e.title,
                "published": e.get("published", ""),
                "link": e.get("link", ""),
            }
            for e in feed.entries[:10]
        ]
    except Exception as e:
        logger.warning(f"RSS fetch failed for {ticker}: {e}")
        headlines = []

    if not headlines:
        return {
            "ticker": ticker,
            "headlines": [],
            "sentiment": "unavailable",
            "summary": "No recent news headlines found for this ticker.",
        }

    # Keyword-based sentiment score
    positive_words = [
        "beat",
        "surpass",
        "growth",
        "strong",
        "record",
        "gain",
        "rises",
        "upgrade",
        "buy",
        "profit",
    ]
    negative_words = [
        "miss",
        "decline",
        "falls",
        "weak",
        "loss",
        "cut",
        "downgrade",
        "sell",
        "risk",
        "lawsuit",
    ]
    all_text = " ".join(h["title"].lower() for h in headlines)
    pos = sum(1 for w in positive_words if w in all_text)
    neg = sum(1 for w in negative_words if w in all_text)
    sentiment = "positive" if pos > neg else "negative" if neg > pos else "neutral"

    # LLM sentiment summary
    headline_text = "\n".join(f"- {h['title']}" for h in headlines)
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Analyze the sentiment of these recent news headlines for {ticker}. "
                f"Provide:\n1) **Overall Sentiment** (Positive / Neutral / Negative) with one-line rationale\n"
                f"2) **Key Themes** - 2-3 bullet points on what the news is focused on\n"
                f"3) **Market Implication** - one sentence on what this signals for investors\n\n"
                f"Headlines:\n{headline_text}"
            )
        ),
    ]
    summary = await generate_section(llm, messages, context="summary")

    return {
        "ticker": ticker,
        "headlines": headlines,
        "sentiment": sentiment,
        "summary": summary,
    }


@app.get("/vector_stores/")
def get_vector_stores():
    stores = faiss_manager.list_stores()
    return {"available_stores": stores, "total_count": len(stores)}


@app.get("/cache/")
def get_cache():
    """List all tickers currently in the in-memory analysis cache."""
    grouped: dict[str, list] = {}
    for key in analysis_cache:
        if ":" not in key:
            continue  # skip bare-ticker aliases
        ticker, form_type, mode = key.split(":", 2)
        grouped.setdefault(ticker, []).append(f"{form_type} ({mode.capitalize()})")
    # Sort each ticker's variants: Executive before Analyst, 10-K before 10-Q
    for ticker in grouped:
        grouped[ticker] = sorted(
            grouped[ticker], key=lambda v: (v.split()[0], 0 if "Executive" in v else 1)
        )
    return {"cached": grouped, "total_tickers": len(grouped)}


@app.get("/vector_store_info/")
def get_vector_store_info(ticker: str = Query(...), form_type: str = Query("10-K")):
    return faiss_manager.get_store_info(ticker, form_type)


@app.get("/")
def root():
    return {
        "message": "FinSight AI - SEC 10-K/10-Q Analysis",
        "version": "2.0.0",
        "provider": get_provider_name(),
        "endpoints": {
            "health": "/health",
            "company_search": "/company_search/?q={q}",
            "analyze_filing": "POST /analyze_filing/?ticker={ticker}",
            "job_status": "/job_status/{job_id}",
            "ask": "/ask/?question={q}&ticker={ticker}&session_id={id}",
            "docs": "/docs",
        },
    }
