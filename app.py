import streamlit as st
import spacy
import json
import re
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from spacy.matcher import Matcher

# ============================
# LOAD MODEL (STREAMLIT SAFE)
# ============================
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Text Intelligence • spaCy",
    page_icon="🧠",
    layout="centered"
)

# ============================
# PREMIUM UI STYLE
# ============================
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 2rem; max-width: 900px; }
h1, h2, h3 { color: #ffffff; }
.card {
    background: #161b22;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
}
.badge {
    display: inline-block;
    background: #238636;
    color: white;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    margin: 2px;
}
.entity {
    background: #1f6feb;
    color: white;
    padding: 4px 8px;
    border-radius: 6px;
    margin: 2px;
    display: inline-block;
}
mark {
    background-color: #f1c40f;
    padding: 2px 4px;
    border-radius: 4px;
}
.small-muted {
    color: #8b949e;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ============================
# SKILL MATCHER
# ============================
skills = [
    "python", "spacy", "nlp",
    "machine learning", "docker",
    "aws", "sql"
]

matcher = Matcher(nlp.vocab)
patterns = [[{"LOWER": tok} for tok in skill.split()] for skill in skills]
matcher.add("SKILLS", patterns)

# ============================
# PDF REPORT
# ============================
def generate_pdf(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>AI Text Intelligence Report</b>", styles["Title"]))
    content.append(Paragraph(f"<b>Text:</b> {data['text']}", styles["Normal"]))
    content.append(Paragraph(
        f"<b>Sentiment:</b> {data['sentiment']} ({data['confidence']}%)",
        styles["Normal"]
    ))
    content.append(Paragraph(
        f"<b>Skills:</b> {', '.join(data['skills']) or 'None'}",
        styles["Normal"]
    ))

    ents = "; ".join([f"{e[0]}: {e[1]}" for e in data["entities"]]) or "None"
    content.append(Paragraph(f"<b>Entities:</b> {ents}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ============================
# HIGHLIGHT ENTITIES
# ============================
def highlight_entities(text, entities):
    highlighted = text
    for label, value in entities:
        highlighted = re.sub(
            rf"\b({re.escape(value)})\b",
            r"<mark>\1</mark>",
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted

# ============================
# SESSION STATE
# ============================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================
# HEADER
# ============================
st.markdown("## 🧠 AI Text Intelligence Dashboard")
st.markdown(
    "<p class='small-muted'>Modern NLP • spaCy • Skills • Entities • Reports</p>",
    unsafe_allow_html=True
)

# ============================
# INPUT CARD
# ============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
text = st.text_area(
    "Enter text for analysis",
    placeholder="Example: I use Python and spaCy for NLP at Google.",
    height=140
)
st.markdown("</div>", unsafe_allow_html=True)

# ============================
# ANALYSIS
# ============================
if st.button("🚀 Analyze Text", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        doc = nlp(text)

        pos_words = {"love", "great", "amazing", "good", "excellent"}
        neg_words = {"bad", "hate", "poor", "terrible"}

        pos = sum(1 for t in doc if t.lemma_.lower() in pos_words)
        neg = sum(1 for t in doc if t.lemma_.lower() in neg_words)

        total = pos + neg if pos + neg else 1
        sentiment = "Positive" if pos >= neg else "Negative"
        confidence = round(max(pos, neg) / total * 100, 2)

        matches = matcher(doc)
        found_skills = sorted(set(doc[s:e].text for _, s, e in matches))
        entities = [(ent.label_, ent.text) for ent in doc.ents]

        result = {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "skills": found_skills,
            "entities": entities
        }
        st.session_state.history.append(result)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Sentiment")
        st.success(f"{sentiment} ({confidence}%)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🛠️ Skills Detected")
        if found_skills:
            for s in found_skills:
                st.markdown(f"<span class='badge'>{s}</span>", unsafe_allow_html=True)
        else:
            st.write("No skills detected")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🏷️ Named Entities")
        if entities:
            for label, value in entities:
                st.markdown(
                    f"<span class='entity'>{label}: {value}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.write("No entities found")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🖍️ Highlighted Text")
        st.markdown(highlight_entities(text, entities), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download JSON",
                json.dumps(result, indent=2),
                "analysis.json",
                "application/json",
                use_container_width=True
            )
        with col2:
            pdf = generate_pdf(result)
            st.download_button(
                "📄 Download PDF",
                pdf,
                "analysis.pdf",
                "application/pdf",
                use_container_width=True
            )

# ============================
# HISTORY
# ============================
if st.session_state.history:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🕘 Recent Analyses")
    for item in reversed(st.session_state.history[-3:]):
        st.markdown(
            f"• **{item['sentiment']} ({item['confidence']}%)** — {item['text'][:80]}..."
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<p class='small-muted'>Portfolio Project • spaCy • Streamlit Cloud</p>",
    unsafe_allow_html=True
)
