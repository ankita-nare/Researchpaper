import os
# Vercel-compatible matplotlib setup is mandatory
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import zipfile
import io
import difflib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import fitz  # PyMuPDF
from habanero import Crossref
from google import genai
from google.genai import types

app = Flask(__name__)

# ── Ethical Guardrails ────────────────────────────────────────────────────────
BLOCKED_PHRASES = [
    "write complete paper", "write the entire paper", "generate full paper",
    "complete paper", "write my paper", "write entire thesis", "write whole paper",
    "do the whole paper", "full paper in one", "entire manuscript",
]

def is_blocked_request(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in BLOCKED_PHRASES)

# ── Gemini Helper ─────────────────────────────────────────────────────────────
def call_gemini(prompt: str, max_tokens: int = 700) -> str:
    """Single call to Gemini using the new google-genai SDK."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API key not configured in environment variables."
    if is_blocked_request(prompt):
        return "🚫 This request has been blocked by ethical guardrails. The assistant cannot generate a complete paper."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"

# ── PDF Extraction ────────────────────────────────────────────────────────────
def extract_pdf_info(pdf_bytes: bytes, filename: str) -> dict:
    """Extract lightweight metadata from PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta = doc.metadata
        text = ""
        for i in range(min(2, len(doc))):
            text += doc[i].get_text()
            
        abstract = ""
        t_lower = text.lower()
        abs_start = t_lower.find("abstract")
        intro_start = t_lower.find("introduction")
        if abs_start != -1:
            end = intro_start if (intro_start != -1 and intro_start > abs_start) else abs_start + 800
            abstract = text[abs_start:end].strip()[:600]
        else:
            abstract = text[:400].strip()
            
        full_text = ""
        for i in range(min(10, len(doc))):
            full_text += doc[i].get_text()
        doc.close()
        
        return {
            "title": meta.get("title") or filename,
            "authors": meta.get("author") or "Unknown",
            "year": meta.get("creationDate", "")[:4] or "Unknown",
            "abstract": abstract,
            "text": full_text[:8000], 
        }
    except Exception as e:
        return {
            "title": filename,
            "authors": "Error",
            "year": "Error",
            "abstract": f"Could not extract: {str(e)}",
            "text": ""
        }

# ── Overleaf / LaTeX Export ───────────────────────────────────────────────────
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

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extract_pdfs", methods=["POST"])
def api_extract_pdfs():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    files = request.files.getlist("files")
    results = []
    for f in files:
        file_bytes = f.read()
        info = extract_pdf_info(file_bytes, f.filename)
        results.append(info)
    return jsonify({"papers": results})

@app.route("/api/generate_gaps", methods=["POST"])
def api_generate_gaps():
    data = request.json
    topic = data.get("topic", "")
    research_type = data.get("research_type", "")
    lit_context = data.get("lit_context", "")
    
    prompt = f"""You are helping an undergraduate student identify research gaps.

Topic: {topic}
Research type: {research_type}
{"Literature summary:" + chr(10) + lit_context if lit_context else "No literature uploaded yet."}

Identify exactly 3 research gaps. For each gap provide:
GAP N: [one sentence description]
BIAS WARNING: [one sentence about potential bias]
EVIDENCE STRENGTH: [Weak / Moderate / Strong]

Keep each gap under 50 words. Be specific and scholarly."""

    result = call_gemini(prompt, max_tokens=500)
    return jsonify({"result": result})

@app.route("/api/generate_thesis", methods=["POST"])
def api_generate_thesis():
    data = request.json
    prompt = f"""You are a thesis writing coach for an undergraduate student.

Topic: {data.get("topic")}
Research Type: {data.get("research_type")}
Research Gap: {data.get("gap")}

Generate exactly 3 thesis statement options. For each:

OPTION [A/B/C]:
Thesis: [one clear, arguable thesis sentence]
Strength: [one sentence]
Weakness: [one sentence]
Researchability: [score 1-10 and one sentence reason]

Keep each option under 80 words total."""

    result = call_gemini(prompt, max_tokens=600)
    return jsonify({"result": result})

@app.route("/api/generate_outline", methods=["POST"])
def api_generate_outline():
    data = request.json
    prompt = f"""Create a short academic paper outline (bullet structure only, no prose).

Topic: {data.get("topic")}
Thesis: {data.get("thesis")}
Research Type: {data.get("research_type")}
Approximate pages: {data.get("page_count")}

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

    result = call_gemini(prompt, max_tokens=400)
    return jsonify({"result": result})

@app.route("/api/draft_section", methods=["POST"])
def api_draft_section():
    data = request.json
    section_choice = data.get("section_choice")
    thesis = data.get("thesis")
    research_type = data.get("research_type")
    outline = data.get("outline", "")
    notes = data.get("notes", "")
    lit_snippet = data.get("lit_snippet", "")
    
    prompt = f"""You are assisting an undergraduate student draft a {section_choice} section.

Paper Thesis: {thesis}
Research Type: {research_type}
{"Outline context: " + outline[:400] if outline else ""}
{"Student notes: " + notes if notes else ""}
{"Relevant literature:" + chr(10) + lit_snippet if lit_snippet else ""}

STRICT RULES:
- Maximum 400 words
- Do NOT invent studies, experiments, data, or statistics
- Only reference literature provided above
- Use hedging language (suggests, may indicate, has been argued)
- Write the {section_choice} only — no other sections
- Academic tone appropriate for an undergraduate paper

Write the {section_choice} section now:"""

    result = call_gemini(prompt, max_tokens=600)
    return jsonify({"result": result})

@app.route("/api/validate_doi", methods=["POST"])
def api_validate_doi():
    doi = request.json.get("doi", "").strip().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
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
                auth_str = "; ".join(f"{a.get('family','')} {a.get('given','')}." for a in authors_raw[:3])
            else:
                auth_str = "Unknown"
            return f'{auth_str} "{title}." *{journal}* {volume}, no. {issue} ({year}): {pages}.'

        return jsonify({
            "doi": doi, "title": title, "authors": authors, "year": year,
            "journal": journal, "volume": volume, "issue": issue, "pages": pages,
            "publisher": publisher, "apa": fmt_apa(), "mla": fmt_mla(), "chicago": fmt_chicago()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/similarity", methods=["POST"])
def api_similarity():
    data = request.json
    text_a = data.get("text", "")
    text_b = data.get("reference_text", "")
    if not text_a or not text_b:
        return jsonify({"score": 0.0})
    score = difflib.SequenceMatcher(None, text_a[:3000], text_b[:3000]).ratio()
    return jsonify({"score": score})

@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.json
    title = data.get("title", "Untitled")
    author = data.get("author", "Author")
    journal = data.get("journal", "Journal")
    template_key = data.get("template", "Basic Article")
    sections = data.get("sections", {})
    outline = data.get("outline", "")
    references = data.get("references", [])

    tmpl = LATEX_TEMPLATES.get(template_key, LATEX_TEMPLATES["Basic Article"])
    body = build_latex_body(sections, outline)
    abstract = sections.get("Introduction", "")[:300] + "..."
    
    try:
        latex = tmpl.format(title=title, author=author, journal=journal, abstract=abstract, body=body)
    except KeyError:
        latex = body

    meta = {
        "title": title, "author": author, "journal": journal, "template": template_key,
        "generated_at": datetime.now().isoformat(), "sections": list(sections.keys()),
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
    return send_file(buf, as_attachment=True, download_name="paper.zip", mimetype="application/zip")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
