"""
AI Research Paper Assistant
A Streamlit MVP to help undergraduate students write theory-based research papers ethically.
Multi-stage generation pipeline with ethical guardrails.
"""

import streamlit as st
import json
import zipfile
import io
import difflib
import re
from datetime import datetime

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from habanero import Crossref
    HABANERO_AVAILABLE = True
except ImportError:
    HABANERO_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

:root {
    --bg-primary: #0f1117;
    --bg-card: #1a1d27;
    --bg-card2: #1f2235;
    --accent: #6c63ff;
    --accent-soft: #4e46b4;
    --green: #22c55e;
    --red: #ef4444;
    --amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #2a2f45;
    --radius: 12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* Warning banner */
.warning-banner {
    background: linear-gradient(90deg, #7c3aed22, #6c63ff22);
    border-left: 4px solid var(--accent);
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 20px;
    font-size: 0.85rem;
    color: #c4b5fd;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 16px;
}

.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--text-primary);
    margin-bottom: 8px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-done { background: #22c55e22; color: #86efac; border: 1px solid #22c55e55; }
.badge-progress { background: #f59e0b22; color: #fcd34d; border: 1px solid #f59e0b55; }
.badge-todo { background: #6b728022; color: #9ca3af; border: 1px solid #6b728055; }
.badge-warning { background: #ef444422; color: #fca5a5; border: 1px solid #ef444455; }
.badge-ok { background: #22c55e22; color: #86efac; border: 1px solid #22c55e55; }

/* Ethics cards */
.ethics-green {
    background: #16a34a18;
    border: 1px solid #22c55e44;
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 12px;
}
.ethics-red {
    background: #dc262618;
    border: 1px solid #ef444444;
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 12px;
}
.ethics-title-green { color: #86efac; font-weight: 700; margin-bottom: 6px; }
.ethics-title-red { color: #fca5a5; font-weight: 700; margin-bottom: 6px; }

/* Section output box */
.output-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 22px;
    white-space: pre-wrap;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-primary);
    margin-top: 12px;
}

/* Disclosure box */
.disclosure-box {
    background: #1e1b4b;
    border: 1px solid #4338ca55;
    border-radius: var(--radius);
    padding: 18px 22px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c7d2fe;
    margin-top: 12px;
}

/* Progress steps */
.step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
}
.step-dot-done { width: 10px; height: 10px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.step-dot-todo { width: 10px; height: 10px; border-radius: 50%; background: var(--border); flex-shrink: 0; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-soft)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--accent) !important;
}

hr { border-color: var(--border) !important; }

.stProgress > div > div { background: var(--accent) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────────
DEFAULTS = {
    "topic": "",
    "title": "",
    "keywords": "",
    "research_type": "Theory Paper",
    "target_journal": "",
    "page_count": 10,
    "word_count": 4000,
    "citation_style": "APA",
    "gap": "",
    "gaps_raw": "",
    "thesis": "",
    "thesis_options": "",
    "outline": "",
    "sections": {},          # {section_name: text}
    "references": [],        # [{doi, title, authors, year, journal, apa, mla, chicago}]
    "uploaded_lit": [],      # [{title, authors, year, abstract, text}]
    "lit_matrix": [],
    "progress": {
        "setup": False,
        "literature": False,
        "gap": False,
        "thesis": False,
        "outline": False,
        "sections": False,
        "citations": False,
    },
    "theme_dark": True,
    "api_key_valid": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Ethical guardrails ────────────────────────────────────────────────────────
BLOCKED_PHRASES = [
    "write complete paper", "write the entire paper", "generate full paper",
    "complete paper", "write my paper", "write entire thesis", "write whole paper",
    "do the whole paper", "full paper in one", "entire manuscript",
]

def is_blocked_request(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in BLOCKED_PHRASES)

# ── Gemini helper ─────────────────────────────────────────────────────────────
def call_gemini(prompt: str, max_tokens: int = 700) -> str:
    """Single call to Gemini Flash. Returns text or raises."""
    if not GENAI_AVAILABLE:
        return "⚠️ google-generativeai package not installed. Run: pip install google-generativeai"
    api_key = st.session_state.get("gemini_api_key", "").strip()
    if not api_key:
        return "⚠️ Please enter your Gemini API key in the sidebar."
    if is_blocked_request(prompt):
        return "🚫 This request has been blocked by ethical guardrails. The assistant cannot generate a complete paper."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf_info(pdf_bytes: bytes, filename: str) -> dict:
    """Extract lightweight metadata from PDF using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        return {
            "title": filename,
            "authors": "Unknown",
            "year": "Unknown",
            "abstract": "PyMuPDF not installed. Run: pip install pymupdf",
            "text": "",
        }
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    meta = doc.metadata
    # Grab first 2 pages as text for abstract extraction
    text = ""
    for i in range(min(2, len(doc))):
        text += doc[i].get_text()
    # Try to get abstract (heuristic: text between "abstract" and "introduction" or first 600 chars)
    abstract = ""
    t_lower = text.lower()
    abs_start = t_lower.find("abstract")
    intro_start = t_lower.find("introduction")
    if abs_start != -1:
        end = intro_start if (intro_start != -1 and intro_start > abs_start) else abs_start + 800
        abstract = text[abs_start:end].strip()[:600]
    else:
        abstract = text[:400].strip()
    # Full text (lightweight: first 10 pages)
    full_text = ""
    for i in range(min(10, len(doc))):
        full_text += doc[i].get_text()
    doc.close()
    return {
        "title": meta.get("title") or filename,
        "authors": meta.get("author") or "Unknown",
        "year": meta.get("creationDate", "")[:4] or "Unknown",
        "abstract": abstract,
        "text": full_text[:8000],  # cap for session state
    }

# ── CrossRef DOI validation ───────────────────────────────────────────────────
def validate_doi(doi: str) -> dict | None:
    """Query CrossRef for a DOI. Returns metadata dict or None."""
    if not HABANERO_AVAILABLE:
        return {"error": "habanero not installed. Run: pip install habanero"}
    doi = doi.strip().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    try:
        cr = Crossref()
        result = cr.works(ids=doi)
        msg = result.get("message", {})
        authors_raw = msg.get("author", [])
        authors = ", ".join(
            f"{a.get('family', '')}, {a.get('given', '')[:1]}." for a in authors_raw[:5]
        ).strip(", ")
        year = ""
        dp = msg.get("published-print") or msg.get("published-online")
        if dp:
            parts = dp.get("date-parts", [[]])
            year = str(parts[0][0]) if parts and parts[0] else ""
        title = msg.get("title", [""])[0]
        journal = (msg.get("container-title") or [""])[0]
        volume = msg.get("volume", "")
        issue = msg.get("issue", "")
        pages = msg.get("page", "")
        publisher = msg.get("publisher", "")

        def fmt_apa():
            a = authors or "Unknown"
            y = f"({year})." if year else ""
            t = f"{title}." if title else ""
            j = f"*{journal}*," if journal else ""
            v = f" *{volume}*" if volume else ""
            i_ = f"({issue})," if issue else ""
            p = f" {pages}." if pages else ""
            return f"{a} {y} {t} {j}{v}{i_}{p} https://doi.org/{doi}"

        def fmt_mla():
            al = authors_raw
            if al:
                first = f"{al[0].get('family','')}, {al[0].get('given','')}"
                rest = ", ".join(f"{a.get('given','')} {a.get('family','')}" for a in al[1:3])
                auth_str = f"{first}{', and ' + rest if rest else ''}"
            else:
                auth_str = "Unknown"
            return f'{auth_str}. "{title}." *{journal}*, vol. {volume}, no. {issue}, {year}, pp. {pages}.'

        def fmt_chicago():
            if authors_raw:
                auth_str = "; ".join(
                    f"{a.get('family','')} {a.get('given','')}." for a in authors_raw[:3]
                )
            else:
                auth_str = "Unknown"
            return f'{auth_str} "{title}." *{journal}* {volume}, no. {issue} ({year}): {pages}.'

        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "publisher": publisher,
            "apa": fmt_apa(),
            "mla": fmt_mla(),
            "chicago": fmt_chicago(),
        }
    except Exception as e:
        return {"error": str(e)}

# ── Similarity check ──────────────────────────────────────────────────────────
def similarity_score(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a[:3000], text_b[:3000]).ratio()

# ── Overleaf / LaTeX export ───────────────────────────────────────────────────
LATEX_TEMPLATES = {
    "IEEE": r"""\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{cite}
\usepackage{amsmath}
\usepackage{hyperref}
\title{{{title}}}
\author{{{author}}}
\begin{document}
\maketitle
\begin{abstract}
{abstract}
\end{abstract}
{body}
\bibliographystyle{{IEEEtran}}
\bibliography{{references}}
\end{document}
""",
    "Springer": r"""\documentclass[twocolumn]{{svjour3}}
\usepackage[utf8]{inputenc}
\usepackage{cite}
\usepackage{hyperref}
\journalname{{{journal}}}
\title{{{title}}}
\author{{{author}}}
\institute{{{author} \at University}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
{abstract}
\end{abstract}
{body}
\bibliographystyle{{spmpsci}}
\bibliography{{references}}
\end{document}
""",
    "APA": r"""\documentclass[12pt,man]{{apa7}}
\usepackage[utf8]{{inputenc}}
\usepackage{{hyperref}}
\title{{{title}}}
\author{{{author}}}
\date{{\today}}
\begin{{document}}
\maketitle
{body}
\printbibliography
\end{{document}}
""",
    "Basic Article": r"""\documentclass[12pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{times}}
\usepackage{{geometry}}
\geometry{{margin=1in}}
\usepackage{{hyperref}}
\title{{{title}}}
\author{{{author}}}
\date{{\today}}
\begin{{document}}
\maketitle
\tableofcontents
\newpage
{body}
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
""",
}

def build_latex_body(sections: dict, outline: str) -> str:
    section_map = {
        "Introduction": "\\section{Introduction}",
        "Literature Review": "\\section{Literature Review}",
        "Methodology": "\\section{Methodology}",
        "Discussion": "\\section{Discussion}",
        "Conclusion": "\\section{Conclusion}",
    }
    body = ""
    for sec, cmd in section_map.items():
        if sec in sections and sections[sec]:
            safe = sections[sec].replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            body += f"{cmd}\n{safe}\n\n"
    return body

def build_bib(references: list) -> str:
    bib = ""
    for i, ref in enumerate(references):
        key = f"ref{i+1}"
        bib += f"@article{{{key},\n"
        bib += f"  title   = {{{ref.get('title','')}}},\n"
        bib += f"  author  = {{{ref.get('authors','')}}},\n"
        bib += f"  journal = {{{ref.get('journal','')}}},\n"
        bib += f"  year    = {{{ref.get('year','')}}},\n"
        bib += f"  doi     = {{{ref.get('doi','')}}},\n"
        bib += "}\n\n"
    return bib

def build_export_zip(title, author, journal, template_key, sections, outline, references) -> bytes:
    tmpl = LATEX_TEMPLATES.get(template_key, LATEX_TEMPLATES["Basic Article"])
    body = build_latex_body(sections, outline)
    abstract = sections.get("Introduction", "")[:300] + "..."
    try:
        latex = tmpl.format(
            title=title or "Untitled",
            author=author or "Author",
            journal=journal or "Journal",
            abstract=abstract,
            body=body,
        )
    except KeyError:
        latex = body

    meta = {
        "title": title,
        "author": author,
        "journal": journal,
        "template": template_key,
        "generated_at": datetime.now().isoformat(),
        "sections": list(sections.keys()),
        "references_count": len(references),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", latex)
        zf.writestr("references.bib", build_bib(references))
        zf.writestr("project_metadata.json", json.dumps(meta, indent=2))
        for sec, text in sections.items():
            safe_name = sec.lower().replace(" ", "_")
            zf.writestr(f"sections/{safe_name}.txt", text or "")
    buf.seek(0)
    return buf.read()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Research Assistant")
    st.markdown("---")

    # API Key
    st.markdown("### 🔑 Gemini API Key")
    api_key_input = st.text_input(
        "Enter your API key",
        type="password",
        key="gemini_api_key",
        help="Get your key at aistudio.google.com",
        label_visibility="collapsed",
        placeholder="AIza...",
    )
    if api_key_input:
        st.markdown('<span class="badge badge-ok">● Key entered</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-warning">● No key</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Research Settings")
    st.session_state.citation_style = st.selectbox(
        "Citation Style", ["APA", "MLA", "Chicago", "IEEE"], index=0
    )
    st.session_state.target_journal = st.text_input(
        "Target Journal", value=st.session_state.target_journal, placeholder="e.g. PLOS ONE"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.page_count = st.number_input("Pages", min_value=4, max_value=40, value=st.session_state.page_count)
    with col2:
        st.session_state.word_count = st.number_input("Words", min_value=1000, max_value=15000, step=500, value=st.session_state.word_count)

    st.markdown("---")
    st.markdown("### 📊 Project Progress")
    progress_labels = {
        "setup": "1. Project Setup",
        "literature": "2. Literature",
        "gap": "3. Research Gap",
        "thesis": "4. Thesis",
        "outline": "5. Outline",
        "sections": "6. Sections",
        "citations": "7. Citations",
    }
    done_count = sum(1 for v in st.session_state.progress.values() if v)
    total = len(st.session_state.progress)
    st.progress(done_count / total)
    st.caption(f"{done_count}/{total} stages complete")
    for k, label in progress_labels.items():
        done = st.session_state.progress.get(k, False)
        dot = "step-dot-done" if done else "step-dot-todo"
        st.markdown(
            f'<div class="step"><div class="{dot}"></div><span style="color:{"#86efac" if done else "#64748b"}">{label}</span></div>',
            unsafe_allow_html=True,
        )

# ── Global warning banner ─────────────────────────────────────────────────────
st.markdown(
    '<div class="warning-banner">⚠️ <strong>AI-generated content must be reviewed and verified before submission.</strong> '
    'This tool assists — it does not write your paper. You are responsible for all content.</div>',
    unsafe_allow_html=True,
)

# ── Main title ─────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:4px'>AI Research Paper Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;margin-bottom:24px'>Guiding you through ethical, structured academic writing — one stage at a time.</p>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Setup",
    "📄 Literature",
    "🔍 Research Gap",
    "🎯 Thesis",
    "📝 Outline",
    "✍️ Section Writer",
    "🔖 Citations",
    "⚖️ Ethics",
    "📦 Export",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PROJECT SETUP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## 📋 Project Setup")
    st.markdown("Fill in the details about your research paper. This helps the AI give you better, more targeted assistance.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.title = st.text_input(
            "Paper Title", value=st.session_state.title,
            placeholder="e.g. The Impact of Social Media on Academic Performance"
        )
        st.session_state.topic = st.text_area(
            "Research Topic (2–4 sentences)", value=st.session_state.topic,
            placeholder="Describe your topic briefly. What is it about? Why does it matter?",
            height=100,
        )
        st.session_state.keywords = st.text_input(
            "Keywords (comma-separated)", value=st.session_state.keywords,
            placeholder="social media, academic performance, undergraduate students"
        )
    with col2:
        st.session_state.research_type = st.selectbox(
            "Research Type",
            ["Theory Paper", "Survey", "Case Study", "Experiment", "Literature Review"],
            index=["Theory Paper", "Survey", "Case Study", "Experiment", "Literature Review"].index(
                st.session_state.research_type
            ),
        )
        author_name = st.text_input("Your Name (for export)", value=st.session_state.get("author_name", ""), placeholder="Jane Smith")
        st.session_state.author_name = author_name
        institution = st.text_input("Institution", value=st.session_state.get("institution", ""), placeholder="University of ...")
        st.session_state.institution = institution

    st.markdown("---")
    if st.button("✅ Save Project Setup", key="save_setup"):
        if not st.session_state.topic or not st.session_state.title:
            st.error("Please fill in at least the Paper Title and Research Topic.")
        else:
            st.session_state.progress["setup"] = True
            st.success("Project saved! Move on to the Literature tab.")

    # Setup summary card
    if st.session_state.progress["setup"]:
        st.markdown('<div class="card"><div class="card-title">Project Summary</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Type", st.session_state.research_type)
        col2.metric("Target Pages", st.session_state.page_count)
        col3.metric("Target Words", f"{st.session_state.word_count:,}")
        st.markdown(f"**Title:** {st.session_state.title}")
        st.markdown(f"**Keywords:** {st.session_state.keywords}")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LITERATURE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📄 Literature Upload")
    st.markdown("Upload PDF papers you've collected. The assistant will extract key information and build a literature matrix.")

    if not PYMUPDF_AVAILABLE:
        st.warning("📦 PyMuPDF is not installed. Install it with: `pip install pymupdf`")

    uploaded_files = st.file_uploader(
        "Upload PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 Extract Literature", key="extract_lit"):
        with st.spinner("Extracting paper metadata..."):
            st.session_state.uploaded_lit = []
            for f in uploaded_files:
                info = extract_pdf_info(f.read(), f.name)
                st.session_state.uploaded_lit.append(info)
            st.session_state.progress["literature"] = True
            st.success(f"Extracted {len(uploaded_files)} papers.")

    if st.session_state.uploaded_lit:
        st.markdown("### 📊 Literature Matrix")
        st.markdown(
            "Review and fill in the **Key Findings**, **Methodology**, and **Possible Gap** columns manually after reading each paper."
        )
        import pandas as pd

        # Build editable dataframe
        rows = []
        for p in st.session_state.uploaded_lit:
            rows.append({
                "Title": p.get("title", "")[:60],
                "Authors": p.get("authors", "")[:40],
                "Year": p.get("year", ""),
                "Abstract Preview": p.get("abstract", "")[:120] + "...",
                "Key Findings": "",
                "Methodology": "",
                "Possible Gap": "",
            })
        df = pd.DataFrame(rows)
        edited_df = st.data_editor(df, use_container_width=True, num_rows="fixed", key="lit_matrix_editor")
        st.session_state.lit_matrix = edited_df.to_dict(orient="records")

        st.markdown("### 📖 Paper Summaries")
        for i, p in enumerate(st.session_state.uploaded_lit):
            with st.expander(f"📄 {p.get('title','Paper ' + str(i+1))[:70]}"):
                st.markdown(f"**Authors:** {p.get('authors','')}")
                st.markdown(f"**Year:** {p.get('year','')}")
                st.markdown(f"**Abstract:**\n{p.get('abstract','')}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RESEARCH GAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🔍 Research Gap Finder")
    st.markdown(
        "Based on your topic and uploaded literature, the AI will identify potential research gaps. "
        "It returns **3 gaps maximum** to keep things focused."
    )

    if not st.session_state.topic:
        st.warning("Please complete Project Setup first.")
    else:
        # Build compact context from literature
        lit_context = ""
        if st.session_state.uploaded_lit:
            for p in st.session_state.uploaded_lit[:5]:
                lit_context += f"- {p.get('title','')} ({p.get('year','')}) by {p.get('authors','')}: {p.get('abstract','')[:200]}\n"

        if st.button("🔍 Generate Research Gaps", key="gen_gaps"):
            prompt = f"""You are helping an undergraduate student identify research gaps.

Topic: {st.session_state.topic}
Research type: {st.session_state.research_type}
{"Literature summary:" + chr(10) + lit_context if lit_context else "No literature uploaded yet."}

Identify exactly 3 research gaps. For each gap provide:
GAP N: [one sentence description]
BIAS WARNING: [one sentence about potential bias]
EVIDENCE STRENGTH: [Weak / Moderate / Strong]

Keep each gap under 50 words. Be specific and scholarly."""

            with st.spinner("Identifying research gaps..."):
                result = call_gemini(prompt, max_tokens=500)
                st.session_state.gaps_raw = result
                st.session_state.progress["gap"] = True

        if st.session_state.gaps_raw:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Identified Research Gaps")
            st.markdown(st.session_state.gaps_raw)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### Select Your Gap")
            gap_input = st.text_area(
                "Paste or type your chosen research gap here:",
                value=st.session_state.gap,
                placeholder="Copy one of the gaps above and refine it in your own words.",
                height=80,
            )
            if st.button("💾 Save Gap", key="save_gap"):
                st.session_state.gap = gap_input
                st.success("Gap saved! Move to Thesis Builder.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — THESIS BUILDER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🎯 Thesis Builder")
    st.markdown(
        "The AI generates **3 thesis options** for your chosen gap. "
        "Pick one, refine it in your own words, and save it."
    )

    if not st.session_state.gap:
        st.warning("Please identify and save a research gap first (Tab 3).")
    else:
        st.markdown(f"**Your Gap:** {st.session_state.gap}")

        if st.button("🎯 Generate Thesis Options", key="gen_thesis"):
            prompt = f"""You are a thesis writing coach for an undergraduate student.

Topic: {st.session_state.topic}
Research Type: {st.session_state.research_type}
Research Gap: {st.session_state.gap}

Generate exactly 3 thesis statement options. For each:

OPTION [A/B/C]:
Thesis: [one clear, arguable thesis sentence]
Strength: [one sentence]
Weakness: [one sentence]
Researchability: [score 1-10 and one sentence reason]

Keep each option under 80 words total."""

            with st.spinner("Generating thesis options..."):
                result = call_gemini(prompt, max_tokens=600)
                st.session_state.thesis_options = result
                st.session_state.progress["thesis"] = True

        if st.session_state.thesis_options:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Thesis Options")
            st.markdown(st.session_state.thesis_options)
            st.markdown('</div>', unsafe_allow_html=True)

            thesis_input = st.text_area(
                "Your Thesis Statement (write it in your own words):",
                value=st.session_state.thesis,
                placeholder="Refine the thesis in your own voice. This is YOUR argument.",
                height=80,
            )
            if st.button("💾 Save Thesis", key="save_thesis"):
                st.session_state.thesis = thesis_input
                st.success("Thesis saved! Move to Outline Builder.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — OUTLINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 📝 Outline Builder")
    st.markdown(
        "Generates a structured academic outline — **no section text** is written here. "
        "You can edit the outline after generation."
    )

    if not st.session_state.thesis:
        st.warning("Please save your thesis statement first (Tab 4).")
    else:
        st.markdown(f"**Your Thesis:** {st.session_state.thesis}")

        if st.button("📝 Generate Outline", key="gen_outline"):
            prompt = f"""Create a short academic paper outline (bullet structure only, no prose).

Topic: {st.session_state.topic}
Thesis: {st.session_state.thesis}
Research Type: {st.session_state.research_type}
Approximate pages: {st.session_state.page_count}

Format:
1. Introduction
   • Background
   • Problem Statement
   • Thesis
2. Literature Review
   • [2-3 thematic sub-headings]
3. Methodology
   • [approach and tools]
4. Discussion
   • [key argument points]
5. Conclusion
   • Summary
   • Future Work

Keep sub-bullets to ONE line each. No prose. No explanations."""

            with st.spinner("Building outline..."):
                result = call_gemini(prompt, max_tokens=400)
                st.session_state.outline = result
                st.session_state.progress["outline"] = True

        if st.session_state.outline:
            edited_outline = st.text_area(
                "Edit your outline here:",
                value=st.session_state.outline,
                height=300,
            )
            if st.button("💾 Save Outline", key="save_outline"):
                st.session_state.outline = edited_outline
                st.success("Outline saved! Move to Section Writer.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — SECTION WRITER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## ✍️ Section Writer")
    st.markdown(
        "Write **one section at a time.** Maximum 400 words per section. "
        "You must review, edit, and verify all output before using it."
    )
    st.markdown(
        '<div class="warning-banner">🚫 <strong>This tool never generates a complete paper.</strong> '
        'Each section must be reviewed by you. Do not submit AI text without verification.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.outline:
        st.warning("Please complete your outline first (Tab 5).")
    else:
        section_choice = st.selectbox(
            "Which section do you want to draft?",
            ["Introduction", "Literature Review", "Methodology", "Discussion", "Conclusion"],
        )

        section_notes = st.text_area(
            f"Your notes for the {section_choice} (what to include, key points):",
            placeholder="Add your own ideas, sources to reference, arguments to make...",
            height=100,
            key=f"notes_{section_choice}",
        )

        # Show lit context selector
        lit_titles = [p.get("title", f"Paper {i+1}") for i, p in enumerate(st.session_state.uploaded_lit)]
        selected_papers = []
        if lit_titles:
            selected_papers = st.multiselect(
                "Reference these uploaded papers (optional):",
                options=lit_titles,
                key=f"papers_{section_choice}",
            )

        if st.button(f"✍️ Draft {section_choice}", key=f"gen_{section_choice}"):
            # Build compact lit snippet
            lit_snippet = ""
            if selected_papers:
                for p in st.session_state.uploaded_lit:
                    if p.get("title","") in selected_papers:
                        lit_snippet += f"- {p.get('title','')} ({p.get('year','')}): {p.get('abstract','')[:200]}\n"

            prompt = f"""You are assisting an undergraduate student draft a {section_choice} section.

Paper Thesis: {st.session_state.thesis}
Research Type: {st.session_state.research_type}
{"Outline context: " + st.session_state.outline[:400] if st.session_state.outline else ""}
{"Student notes: " + section_notes if section_notes else ""}
{"Relevant literature:" + chr(10) + lit_snippet if lit_snippet else ""}

STRICT RULES:
- Maximum 400 words
- Do NOT invent studies, experiments, data, or statistics
- Only reference literature provided above
- Use hedging language (suggests, may indicate, has been argued)
- Write the {section_choice} only — no other sections
- Academic tone appropriate for an undergraduate paper

Write the {section_choice} section now:"""

            with st.spinner(f"Drafting {section_choice}..."):
                result = call_gemini(prompt, max_tokens=600)
                if section_choice not in st.session_state.sections:
                    st.session_state.sections[section_choice] = ""
                st.session_state.sections[section_choice] = result
                st.session_state.progress["sections"] = True

        if section_choice in st.session_state.sections and st.session_state.sections[section_choice]:
            st.markdown("### Generated Draft")
            edited = st.text_area(
                "Edit the draft below before saving:",
                value=st.session_state.sections[section_choice],
                height=250,
                key=f"edit_{section_choice}",
            )
            word_count = len(edited.split())
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"💾 Save {section_choice}", key=f"save_sec_{section_choice}"):
                    st.session_state.sections[section_choice] = edited
                    st.success(f"{section_choice} saved ({word_count} words).")
            with col2:
                st.markdown(f'<span class="badge {"badge-warning" if word_count > 400 else "badge-ok"}">{word_count} words</span>', unsafe_allow_html=True)

            # Quick similarity check
            if st.session_state.uploaded_lit:
                all_lit_text = " ".join(p.get("text","") for p in st.session_state.uploaded_lit)
                sim = similarity_score(edited, all_lit_text)
                if sim > 0.4:
                    st.warning("⚠️ High text similarity to uploaded literature detected. Check for mosaic plagiarism.")
                elif sim > 0.2:
                    st.info("ℹ️ Moderate similarity to literature. Review for paraphrasing quality.")

        # Show all saved sections
        if st.session_state.sections:
            st.markdown("---")
            st.markdown("### 💾 Saved Sections")
            for sec, text in st.session_state.sections.items():
                if text:
                    wc = len(text.split())
                    with st.expander(f"📄 {sec} — {wc} words"):
                        st.markdown(text)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — CITATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("## 🔖 Citation Validator")
    st.markdown(
        "Enter a **DOI** to validate a reference via CrossRef. "
        "Only verified references are added to your project."
    )
    st.markdown(
        '<div class="warning-banner">🚫 <strong>Never fabricate references.</strong> '
        'All citations must be verified real publications.</div>',
        unsafe_allow_html=True,
    )

    if not HABANERO_AVAILABLE:
        st.warning("📦 habanero is not installed. Run: `pip install habanero`")

    doi_input = st.text_input(
        "Enter DOI",
        placeholder="e.g. 10.1038/s41586-021-03819-2",
        help="Find DOIs at doi.org or on the paper's abstract page",
    )

    if st.button("🔍 Validate DOI", key="validate_doi"):
        if not doi_input.strip():
            st.error("Please enter a DOI.")
        else:
            with st.spinner("Looking up reference..."):
                ref_data = validate_doi(doi_input)

            if ref_data and "error" not in ref_data:
                st.session_state.progress["citations"] = True
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### ✅ Valid Reference")
                col1, col2, col3 = st.columns(3)
                col1.metric("Year", ref_data.get("year", "—"))
                col2.metric("Volume", ref_data.get("volume", "—") or "—")
                col3.metric("Pages", ref_data.get("pages", "—") or "—")
                st.markdown(f"**Title:** {ref_data.get('title','')}")
                st.markdown(f"**Authors:** {ref_data.get('authors','')}")
                st.markdown(f"**Journal:** {ref_data.get('journal','')}")

                style = st.session_state.citation_style
                fmt_map = {"APA": "apa", "MLA": "mla", "Chicago": "chicago"}
                fmt_key = fmt_map.get(style, "apa")
                st.markdown(f"**{style} Citation:**")
                st.code(ref_data.get(fmt_key, ""), language="")
                st.markdown('</div>', unsafe_allow_html=True)

                if st.button("➕ Add to My References", key="add_ref"):
                    # Check for duplicate DOI
                    existing_dois = [r.get("doi","") for r in st.session_state.references]
                    if ref_data["doi"] not in existing_dois:
                        st.session_state.references.append(ref_data)
                        st.success("Reference added!")
                    else:
                        st.info("This DOI is already in your reference list.")
            elif ref_data and "error" in ref_data:
                st.error(f"❌ Invalid or not found: {ref_data['error']}")
            else:
                st.error("❌ Could not validate DOI. Check the DOI and try again.")

    # Saved references
    if st.session_state.references:
        st.markdown("---")
        st.markdown("### 📚 My Validated References")
        style = st.session_state.citation_style
        fmt_map = {"APA": "apa", "MLA": "mla", "Chicago": "chicago"}
        fmt_key = fmt_map.get(style, "apa")
        for i, ref in enumerate(st.session_state.references):
            with st.expander(f"{i+1}. {ref.get('title','')[:60]}... ({ref.get('year','')})"):
                st.markdown(f"**{style}:** {ref.get(fmt_key,'')}")
                st.markdown(f"**DOI:** https://doi.org/{ref.get('doi','')}")
                if st.button(f"🗑️ Remove", key=f"remove_ref_{i}"):
                    st.session_state.references.pop(i)
                    st.rerun()

    st.markdown("---")
    st.markdown("### 🔎 Similarity Check")
    st.markdown("Compare any text against your uploaded literature to detect potential mosaic plagiarism.")
    check_text = st.text_area("Paste text to check:", height=120, placeholder="Paste a paragraph from your draft...")
    if st.button("🔎 Run Similarity Check", key="run_sim"):
        if not st.session_state.uploaded_lit:
            st.warning("No literature uploaded. Upload PDFs in the Literature tab first.")
        elif not check_text.strip():
            st.warning("Please enter some text to check.")
        else:
            all_lit = " ".join(p.get("text", "") for p in st.session_state.uploaded_lit)
            sim = similarity_score(check_text, all_lit)
            pct = round(sim * 100, 1)
            if sim > 0.4:
                st.error(f"🔴 High Similarity: {pct}% — Potential mosaic plagiarism detected. Rewrite in your own words.")
            elif sim > 0.2:
                st.warning(f"🟡 Medium Similarity: {pct}% — Review your paraphrasing carefully.")
            else:
                st.success(f"🟢 Low Similarity: {pct}% — Text appears sufficiently original.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — ETHICS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("## ⚖️ Academic Ethics Dashboard")
    st.markdown("Understand what AI assistance is ethical and what crosses the line.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="ethics-green">', unsafe_allow_html=True)
        st.markdown('<div class="ethics-title-green">✅ GREEN ZONE — Ethical AI Use</div>', unsafe_allow_html=True)
        green_items = [
            ("💡 Brainstorming", "Using AI to generate ideas and explore perspectives."),
            ("📋 Outlining", "Getting help structuring your paper's sections."),
            ("✏️ Grammar Help", "Using AI to fix grammar and improve clarity."),
            ("🔖 Citation Formatting", "Auto-formatting verified references in APA/MLA/Chicago."),
            ("🔍 Research Planning", "Identifying gaps, questions, and research direction."),
            ("🗣️ Paraphrasing Aid", "Getting suggestions to rephrase — then rewriting yourself."),
        ]
        for icon_label, desc in green_items:
            st.markdown(f"**{icon_label}**  \n{desc}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ethics-red">', unsafe_allow_html=True)
        st.markdown('<div class="ethics-title-red">🚫 RED ZONE — Academic Misconduct</div>', unsafe_allow_html=True)
        red_items = [
            ("🤖 AI as Co-Author", "AI cannot and must not be listed as an author on any paper."),
            ("📚 Fake References", "Never cite papers that don't exist. CrossRef verify everything."),
            ("🔬 Fabricated Results", "Inventing experimental results or data is research fraud."),
            ("🧪 Invented Experiments", "Describing studies you never conducted is dishonest."),
            ("📄 Unreviewed AI Text", "Submitting AI output without critical review is misconduct."),
            ("📋 Complete Paper Generation", "Asking AI to write your entire paper defeats academic learning."),
        ]
        for icon_label, desc in red_items:
            st.markdown(f"**{icon_label}**  \n{desc}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📢 AI Disclosure Statement")
    st.markdown(
        "Most journals now require authors to disclose AI use. Copy the statement below and include it in your paper."
    )
    disclosure = (
        "Generative AI tools were used for brainstorming, outlining, language refinement, "
        "and citation formatting. All content was reviewed and verified by the authors, "
        "who remain fully responsible for the accuracy and integrity of the final manuscript."
    )
    st.markdown(f'<div class="disclosure-box">{disclosure}</div>', unsafe_allow_html=True)
    st.code(disclosure, language="")

    st.markdown("---")
    st.markdown("### 📜 Ethical Commitments Checklist")
    checks = [
        "I will review all AI-generated content before including it in my paper.",
        "I will verify every citation is a real, published work.",
        "I will not list AI as an author or co-author.",
        "I will not submit AI text without understanding and editing it.",
        "I will not fabricate data, experiments, or study results.",
        "I will include an AI disclosure statement in my final submission.",
        "I understand I am fully responsible for all content in my paper.",
    ]
    all_checked = True
    for c in checks:
        val = st.checkbox(c, key=f"ethics_{c[:20]}")
        if not val:
            all_checked = False
    if all_checked:
        st.success("✅ Excellent! You've confirmed all ethical commitments.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown("## 📦 Export to Overleaf / LaTeX")
    st.markdown(
        "Generate a ready-to-upload `.zip` for Overleaf. Only your **verified references** "
        "and **saved sections** are included."
    )

    col1, col2 = st.columns(2)
    with col1:
        template_choice = st.selectbox(
            "LaTeX Template",
            ["Basic Article", "IEEE", "Springer", "APA"],
        )
        export_author = st.text_input(
            "Author Name(s)", value=st.session_state.get("author_name", ""), placeholder="Jane Smith, John Doe"
        )
    with col2:
        export_title = st.text_input(
            "Paper Title", value=st.session_state.title, placeholder="Your paper title"
        )
        export_journal = st.text_input(
            "Journal / Conference", value=st.session_state.target_journal, placeholder="e.g. PLOS ONE"
        )

    # Preview
    st.markdown("### 📋 Export Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sections Ready", len([v for v in st.session_state.sections.values() if v]))
    col2.metric("Verified References", len(st.session_state.references))
    col3.metric("Template", template_choice)

    if st.session_state.sections:
        st.markdown("**Sections that will be included:**")
        for sec, text in st.session_state.sections.items():
            if text:
                wc = len(text.split())
                st.markdown(f"- ✅ {sec} ({wc} words)")
    else:
        st.info("No sections written yet. Go to Section Writer to draft your paper.")

    st.markdown("---")
    if st.button("📦 Generate Export ZIP", key="gen_zip"):
        if not export_title:
            st.error("Please enter a paper title.")
        elif not st.session_state.sections:
            st.error("No sections to export. Write at least one section first.")
        else:
            with st.spinner("Building your LaTeX package..."):
                zip_bytes = build_export_zip(
                    title=export_title,
                    author=export_author,
                    journal=export_journal,
                    template_key=template_choice,
                    sections=st.session_state.sections,
                    outline=st.session_state.outline,
                    references=st.session_state.references,
                )
            st.success("✅ ZIP package ready!")
            st.download_button(
                label="⬇️ Download paper.zip",
                data=zip_bytes,
                file_name="paper.zip",
                mime="application/zip",
            )
            st.markdown(
                "**To use on Overleaf:** Upload the ZIP at [overleaf.com](https://www.overleaf.com) → "
                "New Project → Upload Project → select `paper.zip`."
            )

    st.markdown("---")
    st.markdown("### 📊 Full Project Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Research Details**")
        st.markdown(f"- **Title:** {st.session_state.title or '—'}")
        st.markdown(f"- **Topic:** {st.session_state.topic[:100] + '...' if len(st.session_state.topic) > 100 else st.session_state.topic or '—'}")
        st.markdown(f"- **Research Gap:** {st.session_state.gap[:80] + '...' if len(st.session_state.gap) > 80 else st.session_state.gap or '—'}")
        st.markdown(f"- **Thesis:** {st.session_state.thesis[:80] + '...' if len(st.session_state.thesis) > 80 else st.session_state.thesis or '—'}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Progress**")
        total_words = sum(len(t.split()) for t in st.session_state.sections.values() if t)
        st.markdown(f"- **Words written:** {total_words:,} / {st.session_state.word_count:,}")
        st.markdown(f"- **References validated:** {len(st.session_state.references)}")
        st.markdown(f"- **Papers uploaded:** {len(st.session_state.uploaded_lit)}")
        done = sum(1 for v in st.session_state.progress.values() if v)
        st.markdown(f"- **Stages complete:** {done}/7")
        st.markdown('</div>', unsafe_allow_html=True)