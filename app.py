import io
import streamlit as st
from PyPDF2 import PdfReader
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time

# -------------------------
# Helper Functions
# --------------------------
def candidate_to_dataframe(resumes, scores):
    data = []
    for idx, (candidate, score) in enumerate(zip(resumes, scores), start=1):
        similarity_percentage = round(float(score) * 100, 2)
        # Dummy placeholder for experience level check:
        # Inject custom CSS for dynamic light/dark mode based on user's system preference
        st.markdown(
            """
            <style>
            @media (prefers-color-scheme: dark) {
            body { background-color: #262730; color: #AAAAAA; }
            .stDataFrame { background-color: #333333; }
            }
            @media (prefers-color-scheme: light) {
            body { background-color: #F0F2F6; color: #333333; }
            .stDataFrame { background-color: #ffffff; }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # (In a real scenario, you might use NLP to extract years of experience)
        experience_level = "Mid-level" if "experience" in candidate.get("text", "").lower() else "Not specified"
        
        # Dummy placeholder for skill gap analysis:
        # (Compare candidate skills with JD-required skills. Here we use similarity as a proxy.)
        skill_gap = "Low" if similarity_percentage > 75 else "High"
        
        # Dummy placeholder for ATS keyword optimization score:
        # (In a complete version, you'd analyze the resume for ATS-friendly keywords.)
        ats_keyword_score = similarity_percentage   # For example purposes
        
        # Determine theme from session state (default to Light)
        theme = st.session_state.get("theme", "Light")
        if theme == "Dark":
            css = """
            <style>
            body { background-color: #262730; color: #AAAAAA; }
            .stDataFrame { background-color: #333333; }
            </style>
            """
        else:
            css = """
            <style>
            body { background-color: #F0F2F6; color: #333333; }
            .stDataFrame { background-color: #ffffff; }
            </style>
            """
        st.markdown(css, unsafe_allow_html=True)
        data.append({
            "Rank": idx,
            "Name": candidate["name"],
            "Similarity Score": round(float(score), 4),
            "Similarity Percentage": similarity_percentage,
            "Experience Level": experience_level,
            "Skill Gap": skill_gap,
            "ATS Keyword Score": ats_keyword_score,
            "Preview": (candidate.get("text") or "")[:200] + "..."
        })
    return pd.DataFrame(data)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# --------------------------
# Load pretrained model
# --------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# --------------------------
# Streamlit Dashboard UI
# --------------------------
st.set_page_config(page_title="Project Nightingale", page_icon="🕊️", layout="wide")

st.title("🕊️ Skill Lens AI – Resume Screener")
st.markdown("### 🚀 Find the best candidates by matching resumes against your job description")

# Sidebar (Dashboard controls)
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    jd = st.text_area("📄 Paste Job Description (JD)", height=200, key="jd_input")
    uploaded_files = st.file_uploader(
        "👤 Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )
    top_k = st.slider("Top candidates to show", 1, 10, 5)

# Action Button
if st.sidebar.button("🔎 Rank Candidates"):
    if not jd:
        st.warning("⚠️ Please paste the Job Description first.")
        st.stop()
    if not uploaded_files:
        st.warning("⚠️ Please upload at least 1 resume PDF.")
        st.stop()

    # Resumes extraction
    resumes = []
    progress_bar = st.progress(0, text="Extracting resumes...")
    for i, f in enumerate(uploaded_files):
        text = extract_text_from_pdf(io.BytesIO(f.read()))
        if len(text) > 50:
            resumes.append({"name": f.name, "text": text})
        progress_bar.progress(int((i+1)/len(uploaded_files)*100), text=f"Processing {f.name}")

    if not resumes:
        st.error("❌ Could not extract text from resumes.")
        st.stop()

    # Encode JD & resumes
    st.info("🔄 Encoding Job Description and Resumes...")
    jd_emb = model.encode(jd, convert_to_tensor=True, normalize_embeddings=True)
    res_embs = model.encode([r["text"] for r in resumes],
                            convert_to_tensor=True, normalize_embeddings=True)

    # Similarity
    cos_scores = util.cos_sim(jd_emb, res_embs)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(resumes)))

    # Candidate Results
    st.success("✅ Ranking Complete!")
    st.subheader("🏆 Top Candidates")

    selected_resumes = [resumes[int(idx)] for idx in top_results.indices]
    scores = [float(score) for score in top_results.values]

    df = candidate_to_dataframe(selected_resumes, scores)
    st.dataframe(df, use_container_width=True)

    # CSV Download
    csv = convert_df_to_csv(df)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

    # Charts
    st.subheader("📊 Candidate Score Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df.set_index("Name")["Similarity Score"])
    with col2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.pie(scores, labels=df["Name"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)