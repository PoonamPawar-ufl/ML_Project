# app.py (Streamlit UI)

import sys
import streamlit as st
import pandas as pd
from sentence_transformers import util
import altair as alt

# Add path to job_matcher.py (adjust if needed)
sys.path.append(r"C:\Users\poona\Desktop\MS_ADS\sem3\ML 2\SmartApply\src")

from job_matcher import clean_text, generate_message, embed_model, summary_model, build_reference_message
from metrics import (
    evaluate_matching_for_one_resume,
    generate_synthetic_labels,
    compute_curve_metrics,
    average_bleu,
)


def main():
    st.set_page_config(page_title="SmartApply", layout="wide")
    st.title("🤖 SmartApply: Automated LinkedIn Messaging for Job Hunters")

    st.markdown(
        """
    SmartApply matches your **resume** to **job postings** and generates **personalized LinkedIn messages**.  
    Currently focused on **Data Scientist**, **ML Engineer**, and **AI Engineer** roles.
    """
    )

    # Sidebar controls for visuals and message tone
    st.sidebar.header("Settings")
    tone = st.sidebar.selectbox("Message tone", ["friendly", "formal", "neutral"], index=0)
    top_n = st.sidebar.slider("Top matches to display", 3, 20, 5)

    # Load dataset directly
    dataset_path = r"C:\Users\poona\Desktop\MS_ADS\sem3\ML 2\SmartApply\data\cleaned_jobs.csv"
    df = pd.read_csv(dataset_path)
    target_roles = ["data scientist", "machine learning engineer", "ml engineer", "ai engineer"]
    df_filtered = df[df["title"].str.lower().str.contains("|".join(target_roles), na=False)].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    st.info(f"Loaded {len(df_filtered)} relevant job postings.")

    # Upload resume from PC
    resume_file = st.file_uploader("📄 Upload your resume (TXT)", type=["txt"])

    if resume_file:
        # Load resume
        resume_text = resume_file.read().decode("utf-8", errors="ignore")
        resume_text_clean = clean_text(resume_text)

        # Summarize resume
        st.info("✍️ Generating resume summary for personalization...")
        resume_summary = summary_model(
            resume_text_clean[:1000], max_length=60, min_length=20, do_sample=False
        )[0]["summary_text"]

        # Prepare job descriptions
        df_filtered["combined_text"] = (
            df_filtered["title"].fillna("") + " " +
            df_filtered["description"].fillna("") + " " +
            df_filtered["skills_desc"].fillna("")
        ).apply(clean_text)

        # Compute embeddings and similarity
        st.info("🧠 Computing semantic similarity between your resume and job postings...")
        resume_emb = embed_model.encode(resume_text_clean, convert_to_tensor=True)
        job_embs = embed_model.encode(df_filtered["combined_text"].tolist(), convert_to_tensor=True)

        similarity_scores = util.cos_sim(resume_emb, job_embs)[0].cpu().numpy()
        df_filtered["similarity_score"] = similarity_scores

        # Rank jobs
        df_ranked = df_filtered.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

        # Synthetic evaluation labels
        relevant_job_ids = generate_synthetic_labels(df_ranked, fraction_top=0.2)
        metrics = evaluate_matching_for_one_resume(df_ranked, relevant_job_ids, k=top_n)

        # ---------- Overall Metrics ----------
        st.subheader("📈 Model Performance Metrics (Synthetic)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision@N", f"{metrics['precision@k']:.3f}")
        col2.metric("Recall@N", f"{metrics['recall@k']:.3f}")
        col3.metric("NDCG@N", f"{metrics['ndcg@k']:.3f}")

        # ---------- Top-N coverage ----------
        top_results = df_ranked.head(top_n).copy()
        total_relevant = len(relevant_job_ids)
        in_top_n = top_results["job_id"].isin(relevant_job_ids).sum()
        coverage = in_top_n / total_relevant if total_relevant > 0 else 0.0

        st.subheader("🎯 Top-N Coverage (Synthetic)")
        cov1, cov2 = st.columns(2)
        cov1.metric("Relevant in Top N", f"{in_top_n} / {total_relevant}")
        cov2.metric("Coverage (%)", f"{coverage * 100:.1f}%")

        # ---------- Precision/Recall/NDCG vs K curve ----------
        st.subheader("📉 Precision / Recall / NDCG vs K")
        curve_df = compute_curve_metrics(df_ranked, relevant_job_ids, max_k=20)
        st.line_chart(curve_df.set_index("k")[["precision", "recall", "ndcg"]])

        # ---------- Visual: Match score bar chart ----------
        st.subheader("📊 Match Score Distribution (Top N)")

                
        avg_scores = (
            top_results
                .groupby("title", as_index=False)["similarity_score"]
                .mean()
                .sort_values("similarity_score", ascending=False)
        )

        st.bar_chart(avg_scores, x="title", y="similarity_score")


        # ---------- Visual: Table of top matches ----------
        st.subheader("📋 Top Match Results (Table)")
        st.dataframe(
            top_results[["job_id", "title", "company_name", "location", "similarity_score"]],
            use_container_width=True
        )

        # ---------- Generated messages (and collect for BLEU) ----------
        st.subheader("🏆 Top Job Matches & Generated Messages")
        generated_messages = []
        for i, row in top_results.iterrows():
            st.markdown(f"### {i+1}. {row['title']} — *{row['company_name']}* ({row['location']})")
            st.write(f"**Match Score:** {row['similarity_score']:.2f}")
            st.write(f"**Description:** {row['description'][:300]}...")
            with st.spinner("✉️ Generating personalized message..."):
                message = generate_message(row, resume_summary, tone=tone)
            generated_messages.append(message)
            st.text_area(
                label="Generated LinkedIn Message",
                value=message,
                height=150,
                key=f"linkedin_msg_{i}"  # unique key for each message
            )
            st.divider()

        # ---------- BLEU metric for message quality ----------
        reference_messages = [build_reference_message(row) for _, row in top_results.iterrows()]
        pairs = list(zip(reference_messages, generated_messages))
        bleu = average_bleu(pairs)

        st.subheader("📝 Message Quality Metric (Synthetic BLEU)")
        st.metric("Average BLEU (Top N)", f"{bleu:.3f}")
        st.caption(
            "BLEU computed using simple template-based reference messages "
            "as synthetic ground truth."
        )


    else:
        st.warning("👆 Please upload your resume (.txt) to begin.")


if __name__ == "__main__":
    main()
