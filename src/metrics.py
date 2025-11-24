# metrics.py
import numpy as np
import pandas as pd
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ---------- Ranking metrics for job-resume matching ----------

def precision_at_k(relevance: List[int], k: int) -> float:
    """relevance: list like [1,0,1,0,...] in ranked order."""
    rel_k = relevance[:k]
    if len(rel_k) == 0:
        return 0.0
    return sum(rel_k) / len(rel_k)


def recall_at_k(relevance: List[int], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    rel_k = relevance[:k]
    return sum(rel_k) / total_relevant


def dcg_at_k(relevance: List[int], k: int) -> float:
    rel_k = np.asarray(relevance[:k])
    if rel_k.size == 0:
        return 0.0
    return np.sum((2 ** rel_k - 1) / np.log2(np.arange(2, rel_k.size + 2)))


def ndcg_at_k(relevance: List[int], k: int) -> float:
    dcg = dcg_at_k(relevance, k)
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    return float(dcg / ideal) if ideal > 0 else 0.0


def evaluate_matching_for_one_resume(df_ranked: pd.DataFrame,
                                     relevant_job_ids,
                                     k: int = 5):
    """
    df_ranked: DataFrame of jobs sorted by similarity, must contain 'job_id'.
    relevant_job_ids: iterable of job_ids marked as relevant (ground truth or synthetic).
    """
    relevant_set = set(relevant_job_ids)
    relevance = [1 if jid in relevant_set else 0 for jid in df_ranked["job_id"].tolist()]

    total_relevant = len(relevant_set)
    p = precision_at_k(relevance, k)
    r = recall_at_k(relevance, k, total_relevant)
    ndcg = ndcg_at_k(relevance, k)

    return {
        "precision@k": p,
        "recall@k": r,
        "ndcg@k": ndcg,
    }


# ---------- Synthetic labels (no manual eval file needed) ----------

def generate_synthetic_labels(df_ranked: pd.DataFrame,
                              fraction_top: float = 0.2):
    """
    Generate synthetic relevance labels:
    - Mark top 'fraction_top' of ranked list as relevant (1), rest as non-relevant (0)
    - Returns a list of job_ids considered "relevant"
    """
    n = len(df_ranked)
    top_n = max(1, int(n * fraction_top))
    relevant_job_ids = df_ranked.head(top_n)["job_id"].tolist()
    return relevant_job_ids


def compute_curve_metrics(df_ranked: pd.DataFrame,
                          relevant_job_ids,
                          max_k: int = 20) -> pd.DataFrame:
    """
    Compute precision, recall, ndcg for k = 1..max_k.
    Returns a DataFrame with columns: k, precision, recall, ndcg.
    """
    relevant_set = set(relevant_job_ids)
    relevance = [1 if jid in relevant_set else 0 for jid in df_ranked["job_id"].tolist()]
    total_relevant = len(relevant_set)

    rows = []
    max_k = min(max_k, len(relevance))
    for k in range(1, max_k + 1):
        p = precision_at_k(relevance, k)
        r = recall_at_k(relevance, k, total_relevant)
        n = ndcg_at_k(relevance, k)
        rows.append({"k": k, "precision": p, "recall": r, "ndcg": n})

    return pd.DataFrame(rows)


# ---------- BLEU coherence metric for generated messages ----------

_smooth = SmoothingFunction().method1


def bleu_for_message_pair(reference: str, candidate: str) -> float:
    """
    Compute BLEU score between a reference message and generated candidate.
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=_smooth))


def average_bleu(pairs):
    """
    pairs: list of (reference_message, generated_message)
    Returns average BLEU over all pairs.
    """
    scores = [bleu_for_message_pair(ref, gen) for ref, gen in pairs]
    return float(np.mean(scores)) if scores else 0.0
