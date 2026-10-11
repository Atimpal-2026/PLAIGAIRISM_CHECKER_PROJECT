
import re
from datetime import datetime
from difflib import SequenceMatcher

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(
    page_title="Plagiarism Checker",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Plagiarism Checker")
st.write(
    "Compare two documents and identify textual similarity."
)
st.info(
    "A similarity score is an indicator, not proof of plagiarism."
)


# -------------------------------
# 2. Text preprocessing
# -------------------------------
def clean_text(text):
    """Normalize text before comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# -------------------------------
# 3. Calculate TF-IDF similarity
# -------------------------------
def calculate_similarity(text1, text2):
    """Return cosine similarity as a percentage."""
    cleaned1 = clean_text(text1)
    cleaned2 = clean_text(text2)

    if not cleaned1 or not cleaned2:
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    try:
        vectors = vectorizer.fit_transform(
            [cleaned1, cleaned2]
        )
        score = cosine_similarity(
            vectors[0:1], vectors[1:2]
        )[0][0]

        return round(float(score) * 100, 2)

    except ValueError:
        # For example, when neither document has
        # usable word tokens.
        return 0.0


# -------------------------------
# 4. Find exact matching sentences
# -------------------------------
def find_matching_sentences(text1, text2):
    """Find sentences appearing in both documents."""
    sentences1 = re.split(r"[.!?]+", text1)
    sentences2 = re.split(r"[.!?]+", text2)

    normalized2 = {
        clean_text(sentence): sentence.strip()
        for sentence in sentences2
        if clean_text(sentence)
    }

    matches = []
    seen = set()

    for sentence in sentences1:
        normalized = clean_text(sentence)

        if (
            normalized
            and normalized in normalized2
            and normalized not in seen
        ):
            matches.append(normalized2[normalized])
            seen.add(normalized)

    return matches


# -------------------------------
# 5. Count words
# -------------------------------
def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


# -------------------------------
# 6. Input section
# -------------------------------
st.subheader("Enter or upload your documents")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Document A")
    file1 = st.file_uploader(
        "Upload Document A (.txt)",
        type=["txt"],
        key="file_a"
    )
    text1 = st.text_area(
        "Or paste Document A here",
        height=220,
        key="text_a"
    )

with col2:
    st.markdown("### Document B")
    file2 = st.file_uploader(
        "Upload Document B (.txt)",
        type=["txt"],
        key="file_b"
    )
    text2 = st.text_area(
        "Or paste Document B here",
        height=220,
        key="text_b"
    )


# Uploaded files take priority over pasted text.
if file1 is not None:
    text1 = file1.getvalue().decode("utf-8-sig", errors="replace")

if file2 is not None:
    text2 = file2.getvalue().decode("utf-8-sig", errors="replace")


# -------------------------------
# 7. Comparison button
# -------------------------------
if st.button("Compare Documents", type="primary"):

    if not text1.strip() or not text2.strip():
        st.error(
            "Please provide text for both documents."
        )

    else:
        similarity = calculate_similarity(text1, text2)
        matches = find_matching_sentences(text1, text2)

        st.divider()
        st.subheader("Comparison Results")

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Text Similarity",
            f"{similarity}%"
        )

        result2.metric(
            "Words in Document A",
            word_count(text1)
        )

        result3.metric(
            "Words in Document B",
            word_count(text2)
        )

        st.progress(
            min(max(int(similarity), 0), 100)
        )

        # This is a display band, not a plagiarism verdict.
        if similarity >= 70:
            st.warning(
                "High textual similarity. Review the matching content."
            )
        elif similarity >= 35:
            st.info(
                "Moderate textual similarity. Review the documents."
            )
        else:
            st.success(
                "Low textual similarity according to this method."
            )

        st.caption(
            "The display bands are illustrative project settings, "
            "not universal academic plagiarism thresholds."
        )

        st.subheader("Exact Matching Sentences")

        if matches:
            st.write(
                f"Found {len(matches)} distinct matching sentence(s):"
            )

            for sentence in matches:
                st.markdown(f"- {sentence}")
        else:
            st.write(
                "No exact matching sentences were found."
            )

        # A secondary comparison method for demonstration.
        sequence_score = round(
            SequenceMatcher(
                None,
                clean_text(text1),
                clean_text(text2)
            ).ratio() * 100,
            2
        )

        st.subheader("Additional Comparison")
        st.write(
            f"SequenceMatcher similarity: {sequence_score}%"
        )

        st.caption(
            "Different algorithms can produce different scores."
        )

        # -------------------------------
        # 8. Generate downloadable report
        # -------------------------------
        report = f"""
PLAGIARISM CHECKER - COMPARISON REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Document A word count: {word_count(text1)}
Document B word count: {word_count(text2)}

TF-IDF cosine similarity: {similarity}%
SequenceMatcher similarity: {sequence_score}%

Number of distinct exact matching sentences: {len(matches)}

Matching sentences:
"""

        if matches:
            report += "\n".join(
                f"- {sentence}" for sentence in matches
            )
        else:
            report += "No exact matching sentences found."

        report += """

NOTE:
Similarity scores are not proof of plagiarism.
Review quotations, citations, common phrases, and context.
"""

        st.download_button(
            "Download Comparison Report",
            data=report,
            file_name="plagiarism_report.txt",
            mime="text/plain"
        )

st.divider()
st.caption(
    "Student Mini Project | Python + Streamlit + scikit-learn"
)
