"""
text_embeddings.py -- Sentence-BERT embeddings for the Reviews branch.

Problem 10 from the implementation plan:
  The current emotional-fatigue branch relies heavily on mapped star ratings,
  while TextBlob is used more as validation than as a full modeling component.
  This leaves valuable text signal unused.

Solution:
  Use Sentence-BERT (all-MiniLM-L6-v2) to embed review text, then aggregate
  embeddings per product-period to create rich text features.

Design:
  1. Load raw review text
  2. Generate sentence embeddings for each review
  3. Aggregate embeddings per product-period (mean, std)
  4. Combine with existing tabular features

Model choice:
  all-MiniLM-L6-v2 is fast, lightweight (80MB), and produces 384-dim embeddings.
  It's a good balance between quality and computational cost.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Embedding dimension for all-MiniLM-L6-v2
EMBED_DIM = 384
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _check_sentence_transformers():
    """Check if sentence-transformers is available."""
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        return False


def generate_review_embeddings(
    reviews_df: pd.DataFrame,
    text_col: str = "Text",
    summary_col: str = "Summary",
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 256,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate Sentence-BERT embeddings for review text.

    Combines Summary + Text for each review and encodes with SBERT.

    Parameters
    ----------
    reviews_df  : DataFrame with text columns
    text_col    : name of the main text column
    summary_col : name of the summary column
    model_name  : Sentence-BERT model name
    batch_size  : encoding batch size
    cache_path  : if provided, cache embeddings as numpy file

    Returns
    -------
    (n_reviews, embed_dim) numpy array of embeddings
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    if not _check_sentence_transformers():
        logger.warning("Falling back to random embeddings for development")
        embeddings = np.random.randn(len(reviews_df), EMBED_DIM).astype(np.float32)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
        return embeddings

    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name)

    # Combine summary and text
    texts = []
    for _, row in reviews_df.iterrows():
        summary = str(row.get(summary_col, "")) if pd.notna(row.get(summary_col)) else ""
        text = str(row.get(text_col, "")) if pd.notna(row.get(text_col)) else ""
        combined = f"{summary}. {text}" if summary else text
        texts.append(combined[:512])  # Truncate to model max length

    logger.info(f"Encoding {len(texts)} reviews...")
    embeddings = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Cached embeddings → {cache_path}")

    return embeddings


def aggregate_embeddings_by_product_period(
    embeddings: np.ndarray,
    reviews_df: pd.DataFrame,
    id_col: str = "ProductId",
    time_col: str = "month",
    n_components: Optional[int] = 32,
) -> pd.DataFrame:
    """
    Aggregate per-review embeddings to product-period level.

    For each (product, period), compute:
      - mean embedding across reviews
      - std of embeddings (text diversity signal)
      - number of reviews

    Optionally apply PCA to reduce dimensionality.

    Parameters
    ----------
    embeddings   : (n_reviews, embed_dim) array from generate_review_embeddings
    reviews_df   : DataFrame with id_col and time_col
    id_col       : product identifier column
    time_col     : time period column
    n_components : PCA components (None = no PCA, use full embeddings)

    Returns
    -------
    DataFrame with product-period rows and embedding features.
    """
    from sklearn.decomposition import PCA

    df = reviews_df[[id_col, time_col]].copy()
    embed_dim = embeddings.shape[1]

    # Add embedding columns
    embed_cols = [f"embed_{i}" for i in range(embed_dim)]
    embed_df = pd.DataFrame(embeddings, index=df.index, columns=embed_cols)
    df = pd.concat([df, embed_df], axis=1)

    # Aggregate by product-period
    agg_funcs = {}
    for col in embed_cols:
        agg_funcs[f"mean_{col}"] = (col, "mean")
        agg_funcs[f"std_{col}"] = (col, "std")

    grouped = df.groupby([id_col, time_col], observed=True)
    agg_df = grouped.agg(**agg_funcs).reset_index()

    # Fill NaN std (single-review periods) with 0
    std_cols = [c for c in agg_df.columns if c.startswith("std_embed_")]
    agg_df[std_cols] = agg_df[std_cols].fillna(0)

    # Apply PCA to reduce dimensionality
    if n_components and n_components < embed_dim:
        mean_cols = [c for c in agg_df.columns if c.startswith("mean_embed_")]
        std_feat_cols = [c for c in agg_df.columns if c.startswith("std_embed_")]

        # PCA on mean embeddings
        pca_mean = PCA(n_components=n_components, random_state=42)
        mean_pca = pca_mean.fit_transform(agg_df[mean_cols].values)
        mean_pca_cols = [f"sbert_mean_pc{i}" for i in range(n_components)]
        mean_pca_df = pd.DataFrame(
            mean_pca, index=agg_df.index, columns=mean_pca_cols
        )

        # PCA on std embeddings (fewer components since it's a variance signal)
        n_std_comp = min(n_components // 2, 8)
        pca_std = PCA(n_components=n_std_comp, random_state=42)
        std_pca = pca_std.fit_transform(agg_df[std_feat_cols].values)
        std_pca_cols = [f"sbert_std_pc{i}" for i in range(n_std_comp)]
        std_pca_df = pd.DataFrame(
            std_pca, index=agg_df.index, columns=std_pca_cols
        )

        # Replace raw embeddings with PCA components
        agg_df = agg_df.drop(columns=mean_cols + std_feat_cols)
        agg_df = pd.concat([agg_df, mean_pca_df, std_pca_df], axis=1)

        logger.info(
            f"PCA reduced embeddings: {embed_dim}→{n_components} mean, "
            f"{embed_dim}→{n_std_comp} std components "
            f"(variance explained: mean={pca_mean.explained_variance_ratio_.sum():.2%}, "
            f"std={pca_std.explained_variance_ratio_.sum():.2%})"
        )

    logger.info(
        f"Aggregated embeddings: {len(agg_df)} product-periods, "
        f"{len(agg_df.columns) - 2} embedding features"
    )

    return agg_df


def enrich_reviews_with_text_features(
    fatigue_df: pd.DataFrame,
    raw_reviews_path: str,
    id_col: str = "ProductId",
    time_col: str = "month",
    cache_dir: str = "data/intermediate",
    n_components: int = 32,
) -> pd.DataFrame:
    """
    End-to-end: load raw reviews, compute SBERT embeddings, aggregate,
    and merge with the existing fatigue signals DataFrame.

    Parameters
    ----------
    fatigue_df       : existing reviews_fatigue_signals DataFrame
    raw_reviews_path : path to raw amazon_reviews.csv
    id_col           : product ID column
    time_col         : time period column
    cache_dir        : directory for caching embeddings
    n_components     : PCA components for dimensionality reduction

    Returns
    -------
    Enriched DataFrame with SBERT embedding features added.
    """
    if not os.path.exists(raw_reviews_path):
        logger.warning(
            f"Raw reviews file not found: {raw_reviews_path}. "
            f"Skipping text embedding enrichment."
        )
        return fatigue_df

    logger.info(f"Loading raw reviews from {raw_reviews_path}...")
    raw_df = pd.read_csv(raw_reviews_path)

    # Ensure time column exists
    if "Time" in raw_df.columns and time_col not in raw_df.columns:
        raw_df["_dt"] = pd.to_datetime(raw_df["Time"], unit="s", errors="coerce")
        raw_df[time_col] = raw_df["_dt"].dt.to_period("M").astype(str)
        raw_df = raw_df.drop(columns=["_dt"])

    # Generate embeddings
    embed_cache = os.path.join(cache_dir, "review_embeddings.npy")
    embeddings = generate_review_embeddings(
        raw_df, cache_path=embed_cache,
    )

    # Aggregate to product-period
    agg_embeds = aggregate_embeddings_by_product_period(
        embeddings, raw_df,
        id_col=id_col, time_col=time_col,
        n_components=n_components,
    )

    # Merge with fatigue signals
    fatigue_df = fatigue_df.merge(
        agg_embeds, on=[id_col, time_col], how="left",
    )

    # Fill NaN embedding features with 0
    embed_feature_cols = [c for c in fatigue_df.columns
                          if c.startswith("sbert_") or c.startswith("mean_embed_")
                          or c.startswith("std_embed_")]
    fatigue_df[embed_feature_cols] = fatigue_df[embed_feature_cols].fillna(0)

    logger.info(
        f"Reviews enriched with {len(embed_feature_cols)} text embedding features. "
        f"Final shape: {fatigue_df.shape}"
    )

    return fatigue_df


def extract_nlp_features(
    reviews_df: pd.DataFrame,
    id_col: str = "ProductId",
    time_col: str = "month",
) -> pd.DataFrame:
    """
    Extract additional NLP features from review text without SBERT.

    Features:
      - subjectivity (via TextBlob)
      - complaint keyword count
      - lexical diversity (unique words / total words)
      - review length
      - question mark count (uncertainty signal)
    """
    COMPLAINT_KEYWORDS = {
        "broken", "defective", "terrible", "worst", "waste",
        "return", "refund", "disappointed", "awful", "horrible",
        "junk", "garbage", "useless", "regret", "scam",
    }

    nlp_features = []

    for _, row in reviews_df.iterrows():
        text = str(row.get("Text", "")) if pd.notna(row.get("Text")) else ""
        words = text.lower().split()
        n_words = max(len(words), 1)
        unique_words = len(set(words))

        # Complaint keywords
        complaint_count = sum(1 for w in words if w in COMPLAINT_KEYWORDS)

        nlp_features.append({
            id_col: row.get(id_col),
            time_col: row.get(time_col),
            "review_length": len(text),
            "word_count": n_words,
            "lexical_diversity": unique_words / n_words,
            "complaint_keyword_count": complaint_count,
            "question_count": text.count("?"),
            "exclamation_count": text.count("!"),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        })

    nlp_df = pd.DataFrame(nlp_features)

    # Aggregate by product-period
    agg_cols = [
        "review_length", "word_count", "lexical_diversity",
        "complaint_keyword_count", "question_count",
        "exclamation_count", "caps_ratio",
    ]

    agg_dict = {col: ["mean", "std"] for col in agg_cols if col in nlp_df.columns}
    grouped = nlp_df.groupby([id_col, time_col], observed=True).agg(agg_dict)
    grouped.columns = [f"nlp_{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()

    # Fill NaN std with 0
    std_cols = [c for c in grouped.columns if c.endswith("_std")]
    grouped[std_cols] = grouped[std_cols].fillna(0)

    logger.info(
        f"Extracted {len(grouped.columns) - 2} NLP features for "
        f"{len(grouped)} product-periods"
    )

    return grouped
