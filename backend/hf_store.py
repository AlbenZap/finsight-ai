"""
HuggingFace Dataset Repo - FAISS Store Persistence

Uploads FAISS index files to a private HF Dataset repo after each build,
and restores them on startup so analysis is fast without rebuilding.

Required env vars:
  HF_TOKEN          - HuggingFace access token (write permission)
  HF_DATASET_REPO   - e.g. "name/finsight-faiss-store"

Repo structure mirrors local faiss_stores/:
  AAPL/AAPL_10K_faiss/index.faiss
  AAPL/AAPL_10K_faiss/index.pkl
  AAPL/AAPL_10K_faiss/metadata.json
  MSFT/MSFT_10K_faiss/...
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_config() -> tuple[str | None, str | None]:
    return os.getenv("HF_TOKEN"), os.getenv("HF_DATASET_REPO")


def is_configured() -> bool:
    token, repo_id = _get_config()
    return bool(token and repo_id)


def ensure_repo_exists() -> bool:
    """Create the HF Dataset repo if it doesn't exist yet."""
    token, repo_id = _get_config()
    if not token or not repo_id:
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        logger.info(f"HF Dataset repo ready: {repo_id}")
        return True
    except Exception as e:
        logger.warning(f"Could not ensure HF repo exists: {e}")
        return False


def upload_store(store_path: Path, ticker: str, form_type: str) -> bool:
    """
    Upload a FAISS store folder to the HF Dataset repo.
    Called after every successful FAISS build - runs in a background thread.

    store_path: e.g. faiss_stores/AAPL/AAPL_10K_faiss/
    Uploaded to repo as: AAPL/AAPL_10K_faiss/
    """
    token, repo_id = _get_config()
    if not token or not repo_id:
        return False

    if not store_path.exists():
        logger.warning(f"FAISS store path does not exist: {store_path}")
        return False

    safe_type = form_type.replace("-", "")
    ticker = ticker.upper()
    path_in_repo = f"{ticker}/{ticker}_{safe_type}_faiss"

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(store_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update FAISS store: {ticker} {form_type}",
        )
        logger.info(f"Uploaded FAISS store to HF: {ticker} ({form_type})")
        return True
    except Exception as e:
        logger.warning(f"HF upload failed for {ticker} ({form_type}): {e}")
        return False


def bidirectional_sync(base_dir: Path) -> dict:
    """
    Two-way sync between local faiss_stores/ and HF Dataset repo:
    - Downloads stores present in HF but missing locally
    - Uploads stores present locally but missing in HF

    Returns {"downloaded": int, "uploaded": int}
    """
    token, repo_id = _get_config()
    if not token or not repo_id:
        logger.info("HF_TOKEN or HF_DATASET_REPO not set - skipping sync")
        return {"downloaded": 0, "uploaded": 0}

    try:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi(token=token)

        # Get list of stores in HF repo
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        repo_stores = {
            f.split("/")[0] + "/" + f.split("/")[1]
            for f in repo_files
            if f.count("/") >= 2 and f.split("/")[1].endswith("_faiss")
        }

        # Get list of local stores
        local_stores = {
            f"{p.parent.name}/{p.name}" for p in base_dir.glob("*/*_faiss") if p.is_dir()
        }

        # Download stores in HF but not local
        stores_to_download = repo_stores - local_stores
        downloaded = 0
        if stores_to_download:
            # Pre-create cache dirs to avoid permission errors on restricted filesystems
            for store_key in stores_to_download:
                (base_dir.resolve() / ".cache" / "huggingface" / "download" / store_key).mkdir(
                    parents=True, exist_ok=True
                )
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(base_dir.resolve()),
                token=token,
                ignore_patterns=["*.gitattributes", ".gitattributes", "README.md"],
            )
            downloaded = len(stores_to_download)
            logger.info(f"Downloaded {downloaded} new store(s) from HF")

        # Upload stores local but not in HF
        stores_to_upload = local_stores - repo_stores
        uploaded = 0
        for store_key in stores_to_upload:
            ticker, store_name = store_key.split("/")
            # Parse form_type from store name e.g. AAPL_10K_faiss -> 10-K
            parts = store_name.replace("_faiss", "").split("_")
            form_type = parts[-1].replace("10K", "10-K").replace("10Q", "10-Q")
            store_path = base_dir / ticker / store_name
            if upload_store(store_path, ticker, form_type):
                uploaded += 1

        logger.info(f"Bidirectional sync complete: {downloaded} downloaded, {uploaded} uploaded")
        return {"downloaded": downloaded, "uploaded": uploaded}

    except Exception as e:
        logger.warning(f"Bidirectional sync failed: {e}")
        return {"downloaded": 0, "uploaded": 0}


def restore_all_stores(base_dir: Path) -> int:
    """
    Download all FAISS stores from HF Dataset repo to local disk on startup.
    Uses snapshot_download which only fetches files not already present.

    Returns number of store directories restored.
    """
    token, repo_id = _get_config()
    if not token or not repo_id:
        logger.info("HF_TOKEN or HF_DATASET_REPO not set - skipping FAISS restore")
        return 0

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(base_dir.resolve()),
            token=token,
            ignore_patterns=["*.gitattributes", ".gitattributes", "README.md"],
        )
        count = sum(1 for p in base_dir.glob("*/*_faiss") if p.is_dir())
        logger.info(f"FAISS stores on disk after HF sync: {count} ({repo_id})")
        return count
    except Exception as e:
        logger.warning(f"Failed to restore FAISS stores from HF: {e}")
        return 0
