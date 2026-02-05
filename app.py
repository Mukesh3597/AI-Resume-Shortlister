import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Text Utilities
# -----------------------------
STOP_WORDS = set([
    "a","an","the","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by",
    "with","from","as","is","are","was","were","be","been","being","this","that","these","those",
    "i","you","he","she","it","we","they","my","your","his","her","our","their","me","him","them",
    "can","could","should","would","may","might","must","will","shall",
    "experience","years","year","skill","skills","project","projects","work","working","responsibility",
    "responsibilities","role","roles","team","teams","good","strong"
])

COMMON_SKILLS = [
    # Programming / CS
    "python","java","c","c++","javascript","sql","mysql","postgresql","mongodb",
    "data structures","algorithms","oops","flask","django","fastapi","streamlit",
    # ML / Data
    "machine learning","deep learning","nlp","computer vision","tensorflow","keras","pytorch",
    "scikit-learn","pandas","numpy","matplotlib",
    "classification","regression","clustering","feature engineering","model deployment",
    # Tools
    "git","github","docker","linux","aws","azure","gcp",
    # Web
    "html","css","react","node","rest api","api",
    # Soft
    "communication","leadership","problem solving"
]


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\+\#\.\-\s]", " ", text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])


def extract_resume_text(uploaded_file) -> str:
    # Streamlit uploaded_file safe read
    file_bytes = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    return ""


def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    corpus = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=7000)
    X = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(sim)


def extract_skill_hits(text: str):
    t = clean_text(text)
    hits = []
    for s in COMMON_SKILLS:
        if s in t:
            hits.append(s)
    return sorted(set(hits))


def jd_resume_keyword_report(resume_text: str, jd_text: str, top_n=15):
    corpus = [jd_text, resume_text]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, stop_words="english")
    X = vec.fit_transform(corpus)
    terms = np.array(vec.get_feature_names_out())

    jd_vec = X[0].toarray().ravel()
    top_idx = jd_vec.argsort()[::-1][:top_n]
    top_terms = [terms[i] for i in top_idx if jd_vec[i] > 0]

    resume_clean = clean_text(resume_text)
    matched = [t for t in top_terms if t in resume_clean]
    missing = [t for t in top_terms if t not in resume_clean]
    return matched, missing


def decision_bucket(score_percent: float) -> str:
    if score_percent >= 70:
        return "Selected"
    if score_percent >= 50:
        return "Maybe"
    return "Rejected"


# -----------------------------
# Highlight (B)
# -----------------------------
def highlight_matches(resume_text: str, keywords, max_lines=10):
    if not resume_text or not keywords:
        return []

    lines = [ln.strip() for ln in resume_text.split("\n") if ln.strip()]
    keywords = [k.strip().lower() for k in keywords if k and k.strip()]

    matched_lines = []
    for line in lines:
        l = line.lower()
        # ignore very short lines (design resumes have many tiny lines)
        if len(line) < 25:
            continue

        for kw in keywords:
            if kw in l:
                matched_lines.append(line)
                break

        if len(matched_lines) >= max_lines:
            break

    return matched_lines


# -----------------------------
# Analyze
# -----------------------------
def analyze_one_resume(file_obj, jd_text: str, strictness: int, top_terms_n: int):
    raw_text = extract_resume_text(file_obj)
    if not raw_text.strip():
        return None, "Could not extract text"

    resume_clean = clean_text(raw_text)
    jd_clean = clean_text(jd_text)

    base_sim = tfidf_similarity(resume_clean, jd_clean)
    score = base_sim * 100.0

    matched_terms, missing_terms = jd_resume_keyword_report(resume_clean, jd_clean, top_n=top_terms_n)

    # A) Smart strictness slider penalty
    penalty = (strictness / 100) * len(missing_terms) * 1.2
    penalty = min(25.0, penalty)  # cap penalty
    score = max(0.0, score - penalty)

    decision = decision_bucket(score)

    resume_skills = extract_skill_hits(resume_clean)
    jd_skills = extract_skill_hits(jd_clean)
    missing_skills = [s for s in jd_skills if s not in resume_skills]

    # B) Matched resume lines (for detail view)
    matched_lines = highlight_matches(raw_text, matched_terms, max_lines=10)

    result = {
        "resume_name": file_obj.name,
        "score_percent": round(score, 1),
        "decision": decision,
        "matched_keywords": ", ".join(matched_terms[:12]),
        "missing_keywords": ", ".join(missing_terms[:12]),
        "skills_found": ", ".join(resume_skills[:18]),
        "skills_missing": ", ".join(missing_skills[:18]),
        "matched_lines": "\n".join(matched_lines),

        # raw + lists for detail view
        "raw_text": raw_text,
        "matched_terms_list": matched_terms,
        "missing_terms_list": missing_terms,
        "resume_skills_list": resume_skills,
        "missing_skills_list": missing_skills,
        "matched_lines_list": matched_lines,
    }
    return result, None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Resume Shortlister (Ranking)", page_icon="📄", layout="wide")

st.title("📄 AI Resume Shortlister (ATS-style) — Ranking Version")
st.caption("Upload multiple resumes + paste JD → get Top ranking list + CSV export + highlights.")

left, right = st.columns(2)

with left:
    resumes = st.file_uploader(
        "Upload Resumes (PDF / DOCX / TXT) — multiple",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    show_text = st.checkbox("Detail view में resume text दिखाओ", value=False)

with right:
    jd = st.text_area("Paste Job Description (JD)", height=220, placeholder="Paste job description here...")

    strictness = st.slider(
        "Strictness (ATS sensitivity)",
        min_value=0,
        max_value=100,
        value=40,
        help="0 = relaxed demo, 100 = very strict ATS"
    )

    top_terms_n = st.slider("Top JD terms (for keyword match)", min_value=10, max_value=30, value=18, step=2)

    if len(jd.split()) < 20:
        st.info("Tip: Better results के लिए proper JD (कम से कम 20 words) paste करो.")

st.divider()

run = st.button("🏆 Rank Resumes", use_container_width=True)

if run:
    if not resumes:
        st.error("कम से कम 1 resume upload करो.")
        st.stop()
    if not jd.strip():
        st.error("Job Description paste करो.")
        st.stop()

    rows = []
    errors = []
    progress = st.progress(0)

    for i, f in enumerate(resumes, start=1):
        res, err = analyze_one_resume(f, jd, strictness, top_terms_n)
        if err:
            errors.append(f"{f.name}: {err}")
        else:
            rows.append(res)
        progress.progress(i / len(resumes))

    if errors:
        st.warning("कुछ files read नहीं हुईं:")
        st.write("\n".join([f"- {e}" for e in errors]))

    if not rows:
        st.error("कोई भी resume analyze नहीं हो पाया. TXT format try करो.")
        st.stop()

    df = pd.DataFrame(rows)
    df_rank = df.sort_values(by=["score_percent"], ascending=False).reset_index(drop=True)
    df_rank.insert(0, "rank", np.arange(1, len(df_rank) + 1))

    st.subheader("🏆 Ranking Results")
    show_cols = ["rank", "resume_name", "score_percent", "decision", "matched_keywords", "skills_missing"]
    st.dataframe(df_rank[show_cols], use_container_width=True, height=320)

    # CSV Download
    export_cols = [
        "rank", "resume_name", "score_percent", "decision",
        "matched_keywords", "missing_keywords", "skills_found", "skills_missing"
    ]
    csv_bytes = df_rank[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV Report",
        data=csv_bytes,
        file_name="resume_ranking_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # Detail View
    st.subheader("🔎 Detail View (Top resume / selected resume)")
    options = df_rank["resume_name"].tolist()
    selected_name = st.selectbox("Select resume", options, index=0)

    row = df_rank[df_rank["resume_name"] == selected_name].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{row['score_percent']}%")
    c2.metric("Decision", row["decision"])
    c3.metric("Rank", int(row["rank"]))

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### ✅ Matched JD Keywords")
        st.write(row["matched_keywords"] if row["matched_keywords"] else "—")

        st.markdown("### 🧩 Skills Found")
        st.write(row["skills_found"] if row["skills_found"] else "—")

        st.markdown("### 🟩 Resume Lines Matching JD")
        if row["matched_lines_list"]:
            for ln in row["matched_lines_list"]:
                st.markdown(f"- ✅ {ln}")
        else:
            st.info("No strong matching lines found (resume design-heavy हो सकता है).")

    with colB:
        st.markdown("### ❌ Missing JD Keywords")
        st.write(row["missing_keywords"] if row["missing_keywords"] else "—")

        st.markdown("### ⚠️ Missing Skills")
        st.write(row["skills_missing"] if row["skills_missing"] else "—")

    if show_text:
        st.divider()
        st.markdown("### 📝 Extracted Resume Text")
        st.text_area("Resume Text", value=row["raw_text"], height=320)
