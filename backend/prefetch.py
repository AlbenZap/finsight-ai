#!/usr/bin/env python3
"""
FinSight AI - Standalone FAISS prefetch script.

Pre-indexes SEC filings for popular tickers so the first user request is instant.
Run this separately from the main app (before or alongside it).

Usage:
    # Default 10 tickers (10-K):
    python prefetch.py

    # Custom tickers:
    python prefetch.py AAPL MSFT TSLA

    # Via Docker Compose:
    docker compose exec backend python prefetch.py
    docker compose exec backend python prefetch.py AAPL MSFT TSLA
"""

import asyncio
import logging
import os
import sys

from edgar import Company, set_identity

from faiss_manager import faiss_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "WMT", "NFLX",
]


async def prefetch_ticker(ticker: str, form_type: str = "10-K") -> None:
    ticker = ticker.upper()
    store_path = faiss_manager._get_store_path(ticker, form_type)

    if store_path.exists():
        logger.info(f"[{ticker}] {form_type} vector store already exists - skipping")
        return

    try:
        logger.info(f"[{ticker}] Fetching {form_type} filing from SEC EDGAR...")
        set_identity(os.environ.get("SEC_IDENTITY", "finsight@example.com"))
        company = Company(ticker)
        filings = company.get_filings(form=form_type)
        if not filings:
            logger.warning(f"[{ticker}] No {form_type} filings found")
            return

        latest = filings[0]
        accession_number = latest.accession_number

        logger.info(f"[{ticker}] Extracting text (accession {accession_number})...")
        content = await asyncio.to_thread(latest.text)

        logger.info(f"[{ticker}] Building vector store ({len(content):,} chars)...")
        vector_store = await asyncio.to_thread(
            faiss_manager.create_vector_store,
            content,
            ticker,
            accession_number,
            form_type,
        )

        if vector_store:
            logger.info(f"[{ticker}] Done - {vector_store.index.ntotal} chunks indexed")
        else:
            logger.error(f"[{ticker}] Vector store creation returned None")

    except Exception as e:
        logger.error(f"[{ticker}] Failed: {e}")


async def main() -> None:
    args = sys.argv[1:]
    form_type = "10-K"
    if "--form" in args:
        idx = args.index("--form")
        form_type = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    tickers = [t.upper() for t in args] if args else DEFAULT_TICKERS
    logger.info(f"Prefetching {len(tickers)} ticker(s) [{form_type}]: {', '.join(tickers)}")

    for ticker in tickers:
        await prefetch_ticker(ticker, form_type)
        await asyncio.sleep(1)  # polite pause between SEC EDGAR requests

    logger.info("Prefetch complete")


if __name__ == "__main__":
    asyncio.run(main())