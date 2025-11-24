# job_matcher.py
# SmartApply: Job → Resume Matching & LinkedIn Message Generator

import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from metrics import (
    evaluate_matching_for_one_resume,
    generate_synthetic_labels,
    average_bleu,
)

# ----------------- Models (loaded at module level) -----------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-large")
summary_model = pipeline("summarization", model="facebook/bart-large-cnn")


# ----------------- Utility functions -----------------

def clean_text(text: str) -> str:
    """
    Basic text cleaning: remove non-letters and extra spaces, lowercase.
    """
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def generate_message(row, resume_summary: str, max_words: int = 120, tone: str = "friendly") -> str:
    """
    Generate a personalized LinkedIn message for a given job row and resume summary.
    tone: 'friendly', 'formal', or 'neutral'
    """
    tone_instructions = {
        "friendly": "Use a warm, approachable tone while staying professional.",
        "formal": "Use a very professional and formal tone.",
        "neutral": "Use a clear, concise, and neutral tone.",
    }
    tone_text = tone_instructions.get(tone, tone_instructions["friendly"])

    prompt = f"""
You are a professional job applicant. Write a {tone} LinkedIn message 
to the recruiter of {row['company_name']} for the {row['title']} role in {row['location']}.
Mention my experience: {resume_summary}.

Guidelines:
- Start with a greeting (e.g., Hi [Recruiter Name])
- {tone_text}
- Be concise, polite, and under {max_words} words
- Do NOT use placeholders like ABC Corp or [Company]

Now write the personalized message:
"""
    response = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.5)
    message = response[0]["generated_text"].strip()

    # Limit words
    words = message.split()
    if len(words) > max_words:
        message = " ".join(words[:max_words]) + "..."
    return message


def build_reference_message(row) -> str:
    """
    Simple deterministic template for BLEU comparison.
    """
    return (
        f"Hi, I am interested in the {row['title']} role at "
        f"{row['company_name']} in {row['location']}. "
        f"I would like to connect and learn more about this opportunity."
    )


# ----------------- Example script workflow -----------------

if __name__ == "__main__":
    # Example usage with local files
    df_path = "C:/Users/poona/Desktop/MS_ADS/sem3/ML 2/SmartApply/data/cleaned_jobs.csv"
    resume_path = "C:/Users/poona/Desktop/MS_ADS/sem3/ML 2/SmartApply/data/Poonam_Kishor_Pawar_Resume.txt"

    # Load dataset
    df = pd.read_csv(df_path)
    target_roles = [
        "data scientist", "machine learning engineer",
        "ml engineer", "ai engineer", "artificial intelligence engineer"
    ]
    df_filtered = df[df["title"].str.lower().str.contains("|".join(target_roles), na=False)].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    print(f"{len(df_filtered)} relevant postings found.")

    # Load resume safely
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_text = f.read()
    except UnicodeDecodeError:
        with open(resume_path, "r", encoding="utf-8", errors="ignore") as f:
            resume_text = f.read()

    resume_text_clean = clean_text(resume_text)

    # Summarize resume
    resume_summary = summary_model(
        resume_text_clean[:1000], max_length=60, min_length=20, do_sample=False
    )[0]["summary_text"]

    # Prepare job text
    df_filtered["combined_text"] = (
        df_filtered["title"].fillna("") + " " +
        df_filtered["description"].fillna("") + " " +
        df_filtered["skills_desc"].fillna("")
    ).apply(clean_text)

    # Compute embeddings
    resume_emb = embed_model.encode(resume_text_clean, convert_to_tensor=True)
    job_embs = embed_model.encode(df_filtered["combined_text"].tolist(), convert_to_tensor=True)

    # Similarity scores
    similarity_scores = util.cos_sim(resume_emb, job_embs)[0]
    df_filtered["similarity_score"] = similarity_scores.cpu().numpy()

    # Rank
    df_ranked = df_filtered.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)
    top_jobs = df_ranked.head(5).copy()

    # Generate messages
    top_jobs["personalized_message"] = [
        generate_message(row, resume_summary) for _, row in top_jobs.iterrows()
    ]

    # Display messages
    for i, row in top_jobs.iterrows():
        print(f"\n--- Message {i+1} ({row['title']} at {row['company_name']}) ---")
        print(row["personalized_message"])
        print("\n----------------------")

    # ----------------- Quantitative Evaluation -----------------

    print("\n=== Synthetic Ranking Evaluation (k=5) ===")
    # Generate synthetic labels: top 20% as relevant
    relevant_job_ids = generate_synthetic_labels(df_ranked, fraction_top=0.2)
    metrics = evaluate_matching_for_one_resume(df_ranked, relevant_job_ids, k=5)

    print(f"Precision@5: {metrics['precision@k']:.3f}")
    print(f"Recall@5:    {metrics['recall@k']:.3f}")
    print(f"NDCG@5:      {metrics['ndcg@k']:.3f}")

    # BLEU evaluation for message quality using simple reference templates
    print("\n=== Message Quality (BLEU, top 5) ===")
    reference_messages = [build_reference_message(row) for _, row in top_jobs.iterrows()]
    generated_messages = top_jobs["personalized_message"].tolist()
    pairs = list(zip(reference_messages, generated_messages))
    avg_bleu = average_bleu(pairs)
    print(f"Average BLEU over {len(pairs)} messages: {avg_bleu:.3f}")
