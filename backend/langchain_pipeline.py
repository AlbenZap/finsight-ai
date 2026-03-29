"""
LangChain LLM Pipeline for FinSight AI

Dual-provider setup: Ollama (self-hosted Llama 3.1 8B) or NVIDIA NIM (cloud API).
Switch providers by setting LLM_PROVIDER env var.
"""

import logging
import os

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior financial analyst specializing in SEC filings.

CRITICAL — numbers rule: Every dollar amount, percentage, ratio, and count you write must appear verbatim in the provided context excerpts. Do NOT calculate, derive, estimate, or infer any figure — not even growth rates or margins. If an exact figure is not present verbatim in the context, omit that data point entirely — do not write a placeholder.

CRITICAL — units rule: The denomination used varies by company and is stated in a note at the top of each financial table (e.g., "in thousands, except per share data" or "in millions, except per share data"). Always find and apply the correct denomination from the filing. For thousands-denominated filings: 176,529 = $176.5M or $0.2B. For millions-denominated filings: 34,550 = $34,550M or $34.6B. Never output a bare number without converting it to the appropriate M or B scale.

Always respond in well-structured markdown:
- Use **bold** for all key metrics, dollar figures, and percentages (e.g., **Revenue: $394.3B (+8% YoY)**)
- Use bullet points (`-`) or numbered lists for multi-item content; never write walls of plain text
- Use `###` subheadings to separate distinct topics when a response covers multiple areas
- ALWAYS cite the source section wrapped in markdown italics at the end of each bullet: *Item 1 - Business*, *Item 1A - Risk Factors*, *Item 7 - MD&A*, *Item 8 - Financial Statements*, *Note 13*, etc. — never write a plain-text citation
- ABSOLUTE RULE: write only positive facts that are present in the context. Never write a sentence whose purpose is to report absence. This means zero sentences containing: "not explicitly stated", "not mentioned", "not disclosed", "not addressed", "None explicitly named", "No specific", "does not mention", "is not stated", "are not explicitly", "not available", "not provided", or any equivalent. If a sub-section would contain only absence statements, drop the entire sub-section heading and leave no trace of it.
- Do not use magnitude qualifiers (slight, modest, significant, notable, substantial, marginal) when describing year-over-year changes — state the dollar figures and let the reader judge magnitude
- No raw HTML, no excessive blank lines, no redundant preamble like "Based on the filing..." - go straight to the analysis"""

_llm_cache = None


def get_llm():
    """
    Get or create cached LLM instance.

    Provider selection via LLM_PROVIDER env var:
      - "ollama" (default): self-hosted Llama 3.1 8B via Ollama
      - "nvidia_nim": NVIDIA NIM cloud API (meta/llama-3.1-8b-instruct)

    Both providers support .ainvoke() and .astream().
    """
    global _llm_cache

    if _llm_cache is not None:
        return _llm_cache

    provider = os.getenv("LLM_PROVIDER", "ollama")
    logger.info(f"Initializing LLM provider: {provider}")

    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                temperature=0.1,
            )
        else:
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "NVIDIA_API_KEY environment variable is required when LLM_PROVIDER=nvidia_nim"
                )
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            llm = ChatNVIDIA(
                model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
                api_key=api_key,
                temperature=0.1,
            )

        _llm_cache = llm
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize LLM ({provider}): {e}")
        raise RuntimeError(f"Could not initialize LLM: {e}")


def get_provider_name() -> str:
    """Return human-readable provider name for the UI badge."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return f"Llama 3.1 · Ollama ({model})"
    model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct").split("/")[-1]
    return f"Llama 3.1 8B · NVIDIA NIM ({model})"
