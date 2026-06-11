import os
import streamlit as st

from src.ingestion.pdf_loader import PDFLoader
from src.chunking.sentence_chunker import SentenceChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.faiss_indexer import FAISSIndexer
from src.retrieval.retriever import Retriever
from src.rag.rag_pipeline import RAGPipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:          #080B12;
    --surface:     #0F1320;
    --surface2:    #151929;
    --border:      #1C2236;
    --border2:     #232840;
    --accent:      #3B82F6;
    --accent-dim:  rgba(59,130,246,0.12);
    --accent-glow: rgba(59,130,246,0.25);
    --teal:        #2DD4BF;
    --teal-dim:    rgba(45,212,191,0.10);
    --amber:       #F59E0B;
    --red:         #F87171;
    --text:        #E2E8F6;
    --subtext:     #6B7A99;
    --muted:       #3A4460;
    --mono:        'DM Mono', monospace;
    --sans:        'Outfit', sans-serif;
    --r-sm: 8px; --r-md: 12px; --r-lg: 16px;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1.25rem !important; }
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }

/* brand */
.brand { display:flex; align-items:center; gap:10px; margin-bottom:1.75rem; padding-bottom:1.25rem; border-bottom:1px solid var(--border); }
.brand-icon { width:38px; height:38px; background:linear-gradient(135deg,var(--accent),var(--teal)); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0; }
.brand-name { font-family:var(--sans); font-size:1.15rem; font-weight:800; letter-spacing:-0.02em; color:var(--text); }
.brand-tag  { font-family:var(--mono); font-size:0.6rem; color:var(--accent); background:var(--accent-dim); border:1px solid rgba(59,130,246,0.25); border-radius:4px; padding:2px 6px; }
.sb-label   { font-size:0.65rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:var(--muted); margin-bottom:0.65rem; margin-top:1.25rem; }

/* pipeline steps */
.pipeline-steps { display:flex; flex-direction:column; }
.pipeline-step  { display:flex; align-items:center; gap:10px; padding:7px 0; position:relative; }
.pipeline-step:not(:last-child)::after { content:''; position:absolute; left:10px; top:28px; width:1px; height:calc(100% - 4px); background:var(--border2); }
.step-dot { width:21px; height:21px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.65rem; font-weight:700; flex-shrink:0; z-index:1; }
.step-dot.done { background:var(--teal);    color:#000; }
.step-dot.idle { background:var(--border2); color:var(--muted); }
.step-label      { font-size:0.82rem; font-weight:500; color:var(--subtext); }
.step-label.done { color:var(--teal); }

/* doc pill */
.doc-pill      { display:flex; align-items:flex-start; gap:10px; background:var(--surface2); border:1px solid var(--border2); border-radius:var(--r-md); padding:10px 12px; margin-bottom:7px; }
.doc-pill-name { font-size:0.85rem; font-weight:600; color:var(--text); }
.doc-pill-meta { font-family:var(--mono); font-size:0.68rem; color:var(--muted); margin-top:3px; }

/* page header */
.page-header { display:flex; align-items:flex-end; justify-content:space-between; padding:1.5rem 0 1.75rem; border-bottom:1px solid var(--border); margin-bottom:1.75rem; }
.page-title  { font-family:var(--sans); font-size:1.65rem; font-weight:800; letter-spacing:-0.03em; color:var(--text); line-height:1; }
.page-title span { background:linear-gradient(90deg,var(--accent),var(--teal)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.page-sub    { font-size:0.85rem; color:var(--subtext); margin-top:5px; }
.status-badge { font-family:var(--mono); font-size:0.72rem; padding:5px 12px; border-radius:99px; display:flex; align-items:center; gap:6px; }
.status-badge.ready { background:var(--teal-dim); color:var(--teal); border:1px solid rgba(45,212,191,0.25); }
.status-badge.idle  { background:var(--accent-dim); color:var(--subtext); border:1px solid var(--border2); }

/* stat cards */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:1.75rem; }
.stat-card  { background:var(--surface); border:1px solid var(--border); border-radius:var(--r-lg); padding:1.1rem 1.25rem; position:relative; overflow:hidden; }
.stat-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.stat-card.blue::before  { background:var(--accent); }
.stat-card.teal::before  { background:var(--teal); }
.stat-card.amber::before { background:var(--amber); }
.stat-card.red::before   { background:var(--red); }
.stat-eyebrow { font-family:var(--mono); font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; }
.stat-card.blue  .stat-eyebrow { color:var(--accent); }
.stat-card.teal  .stat-eyebrow { color:var(--teal); }
.stat-card.amber .stat-eyebrow { color:var(--amber); }
.stat-card.red   .stat-eyebrow { color:var(--red); }
.stat-value { font-family:var(--sans); font-size:2rem; font-weight:800; letter-spacing:-0.04em; color:var(--text); line-height:1.1; margin:6px 0 4px; }
.stat-desc  { font-size:0.78rem; color:var(--subtext); }

/* file uploader */
[data-testid="stFileUploader"] section { background:var(--surface) !important; border:1.5px dashed var(--border2) !important; border-radius:var(--r-lg) !important; padding:1.5rem !important; }
[data-testid="stFileUploader"] section:hover { border-color:var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] button { background:var(--accent-dim) !important; border:1px solid rgba(59,130,246,0.3) !important; border-radius:var(--r-sm) !important; color:var(--accent) !important; font-weight:600 !important; }

/* text input */
[data-testid="stTextInput"] > label { font-family:var(--mono) !important; font-size:0.68rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:var(--subtext) !important; }
[data-testid="stTextInput"] input { background:var(--surface) !important; border:1.5px solid var(--border2) !important; border-radius:var(--r-md) !important; color:var(--text) !important; font-family:var(--sans) !important; font-size:0.95rem !important; padding:0.75rem 1rem !important; }
[data-testid="stTextInput"] input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-glow) !important; }
[data-testid="stTextInput"] input::placeholder { color:var(--muted) !important; }

/* buttons */
[data-testid="stButton"] > button { background:var(--accent) !important; color:#fff !important; border:none !important; border-radius:var(--r-md) !important; font-family:var(--sans) !important; font-size:0.9rem !important; font-weight:700 !important; padding:0.65rem 1.75rem !important; transition:background 0.2s,transform 0.1s !important; }
[data-testid="stButton"] > button:hover { background:#2563EB !important; transform:translateY(-1px) !important; }

/* answer */
.answer-wrap   { background:var(--surface); border:1px solid var(--border2); border-radius:var(--r-lg); overflow:hidden; margin-bottom:1.5rem; }
.answer-header { display:flex; align-items:center; gap:10px; padding:0.85rem 1.25rem; border-bottom:1px solid var(--border); background:var(--surface2); }
.answer-header-dot   { width:8px; height:8px; background:var(--teal); border-radius:50%; }
.answer-header-label { font-family:var(--mono); font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--teal); }
.answer-header-query { margin-left:auto; font-size:0.78rem; color:var(--subtext); font-style:italic; max-width:55%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.answer-body { padding:1.5rem; font-size:0.97rem; line-height:1.8; color:var(--text); }

/* sources */
.sources-header { font-family:var(--mono); font-size:0.68rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); margin-bottom:0.8rem; }
.source-card { background:var(--surface); border:1px solid var(--border2); border-radius:var(--r-md); padding:1rem 1.25rem; margin-bottom:10px; position:relative; overflow:hidden; }
.source-card::before { content:''; position:absolute; left:0; top:0; bottom:0; width:3px; background:linear-gradient(180deg,var(--accent),var(--teal)); }
.source-meta  { display:flex; align-items:center; gap:10px; margin-bottom:0.6rem; }
.source-id    { font-family:var(--mono); font-size:0.72rem; background:var(--accent-dim); color:var(--accent); border:1px solid rgba(59,130,246,0.2); border-radius:5px; padding:2px 8px; }
.source-score { font-family:var(--mono); font-size:0.72rem; background:var(--teal-dim); color:var(--teal); border:1px solid rgba(45,212,191,0.2); border-radius:5px; padding:2px 8px; }
.source-text  { font-size:0.83rem; color:var(--subtext); line-height:1.65; border-top:1px solid var(--border); padding-top:0.6rem; }

/* empty state */
.empty-state { text-align:center; padding:3.5rem 2rem; border:1.5px dashed var(--border2); border-radius:var(--r-lg); background:var(--surface); }
.empty-icon  { font-size:2.5rem; margin-bottom:0.75rem; }
.empty-title { font-size:1rem; font-weight:600; color:var(--text); margin-bottom:6px; }
.empty-sub   { font-size:0.85rem; color:var(--subtext); }

[data-testid="stAlert"] { background:var(--teal-dim) !important; border:1px solid rgba(45,212,191,0.25) !important; border-radius:var(--r-md) !important; }
[data-testid="stExpander"] { background:var(--surface) !important; border:1px solid var(--border2) !important; border-radius:var(--r-md) !important; }
[data-testid="stProgressBar"] > div { background:var(--border) !important; border-radius:99px !important; }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,var(--accent),var(--teal)) !important; border-radius:99px !important; }
hr { border-color:var(--border) !important; margin:1.5rem 0 !important; }
@media (max-width:900px) { .stat-grid { grid-template-columns:repeat(2,1fr); } }
</style>
""", unsafe_allow_html=True)


# ── Cache pipeline build (keyed by filename) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_pipeline(pdf_path: str):
    loader     = PDFLoader(pdf_path)
    text       = loader.extract_text()
    chunker    = SentenceChunker()
    chunks     = chunker.chunk_text(text)
    emb_gen    = EmbeddingGenerator()
    embeddings = emb_gen.generate_embeddings(chunks)
    indexer    = FAISSIndexer()
    indexer.build_index(embeddings)
    retriever  = Retriever(emb_gen, indexer, chunks)
    pipeline   = RAGPipeline(retriever)
    return pipeline, text, chunks


# ── Session-state defaults ─────────────────────────────────────────────────────
for _k, _v in {
    "pipeline":        None,
    "doc_meta":        {},
    "history":         [],
    "last_n_results":  "—",
    "last_top_score":  "—",
    "pending_pdf":     None,   # path written to disk, waiting to be indexed
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Step 1 — write uploaded bytes to disk (before any rerun) ──────────────────
# We do this OUTSIDE columns so the file persists across reruns.
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF", type=["pdf"], label_visibility="collapsed",
    key="pdf_uploader"
)

if uploaded_file is not None:
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Only re-index if it's a new file
    current_name = st.session_state["doc_meta"].get("name", "")
    if uploaded_file.name != current_name:
        st.session_state["pending_pdf"] = pdf_path
        st.session_state["pipeline"]    = None   # invalidate old index


# ── Step 2 — run pipeline if pending ──────────────────────────────────────────
if st.session_state["pending_pdf"] and st.session_state["pipeline"] is None:
    _path = st.session_state["pending_pdf"]
    _prog = st.progress(0, text="Loading PDF…")
    _prog.progress(15, text="Extracting text…")
    _pipeline, _text, _chunks = build_pipeline(_path)
    _prog.progress(70, text="Generating embeddings & building index…")
    _prog.progress(100, text="Index ready ✓")
    _prog.empty()

    st.session_state["pipeline"]       = _pipeline
    st.session_state["doc_meta"]       = {
        "name":   os.path.basename(_path),
        "chars":  len(_text),
        "chunks": len(_chunks),
    }
    st.session_state["pending_pdf"]    = None
    st.session_state["last_n_results"] = "—"
    st.session_state["last_top_score"] = "—"
    st.success(f"✓  **{os.path.basename(_path)}** indexed — {len(_chunks)} chunks ready.")


# ── Sidebar (rest) ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-name">DocMind</div>
            <div class="brand-tag">RAG · FAISS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # pipeline steps
    is_ready = st.session_state["pipeline"] is not None
    st.markdown('<div class="sb-label">Pipeline Status</div>', unsafe_allow_html=True)
    steps = [
        ("Load PDF",          is_ready),
        ("Sentence Chunking", is_ready),
        ("Embed Chunks",      is_ready),
        ("Build FAISS Index", is_ready),
        ("Ready to Query",    is_ready),
    ]
    html = '<div class="pipeline-steps">'
    for i, (lbl, done) in enumerate(steps):
        cls  = "done" if done else "idle"
        icon = "✓"    if done else str(i + 1)
        html += f'<div class="pipeline-step"><div class="step-dot {cls}">{icon}</div><div class="step-label {cls}">{lbl}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # upload zone label
    st.markdown('<div class="sb-label">Upload Document</div>', unsafe_allow_html=True)
    # (the actual file_uploader widget was rendered above — shown here via label)

    if st.session_state["doc_meta"]:
        meta = st.session_state["doc_meta"]
        st.markdown(f"""
        <div class="doc-pill">
            <div style="font-size:1.1rem;">📄</div>
            <div>
                <div class="doc-pill-name">{meta['name']}</div>
                <div class="doc-pill-meta">{meta['chars']:,} chars · {meta['chunks']} chunks</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k_sb     = st.slider("Top-K Chunks", 1, 10, 5, key="top_k_sidebar")
    score_thresh = st.slider("Min Score",    0.0, 1.0, 0.0, 0.05, key="score_thresh")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:var(--subtext);line-height:1.7;">
        PDF → sentence chunks →<br>sentence-transformer embeddings →<br>FAISS ANN index → LLM answer<br><br>
        <span style="color:var(--accent)">FAISS</span> &nbsp;·&nbsp;
        <span style="color:var(--teal)">LangChain</span> &nbsp;·&nbsp;
        <span style="color:var(--amber)">Streamlit</span>
    </div>""", unsafe_allow_html=True)


# ── Main header ────────────────────────────────────────────────────────────────
status_cls   = "ready" if is_ready else "idle"
status_label = "● Index Ready" if is_ready else "○ Awaiting Document"
chars  = st.session_state["doc_meta"].get("chars",  0)
chunks = st.session_state["doc_meta"].get("chunks", 0)

st.markdown(f"""
<div class="page-header">
    <div>
        <div class="page-title">Technical Document <span>RAG</span></div>
        <div class="page-sub">Retrieval-Augmented Generation · Semantic Search · FAISS Vector Index</div>
    </div>
    <div class="status-badge {status_cls}">{status_label}</div>
</div>
<div class="stat-grid">
    <div class="stat-card blue">
        <div class="stat-eyebrow">Characters</div>
        <div class="stat-value">{f"{chars:,}" if chars else "—"}</div>
        <div class="stat-desc">Raw text extracted</div>
    </div>
    <div class="stat-card teal">
        <div class="stat-eyebrow">Chunks</div>
        <div class="stat-value">{chunks if chunks else "—"}</div>
        <div class="stat-desc">Sentence segments indexed</div>
    </div>
    <div class="stat-card amber">
        <div class="stat-eyebrow">Results</div>
        <div class="stat-value">{st.session_state['last_n_results']}</div>
        <div class="stat-desc">Sources retrieved last query</div>
    </div>
    <div class="stat-card red">
        <div class="stat-eyebrow">Top Score</div>
        <div class="stat-value">{st.session_state['last_top_score']}</div>
        <div class="stat-desc">Highest similarity score</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.05, 1], gap="large")

# ── Initialise query state that both columns need ──────────────────────────────
ask_clicked  = False
query        = st.session_state.get("active_query", "")
top_k_inline = top_k_sb   # default; overridden inside col_left if rendered


# ═══════════════════════════════ LEFT ═════════════════════════════════════════
with col_left:
    st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:var(--subtext);margin-bottom:0.75rem;">📂  Document Ingestion</div>', unsafe_allow_html=True)

    if not is_ready:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📄</div>
            <div class="empty-title">No document loaded</div>
            <div class="empty-sub">Use the sidebar uploader to index a PDF and start querying.</div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Query panel ──
        st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:var(--subtext);margin-bottom:0.75rem;">🔍  Ask a Question</div>', unsafe_allow_html=True)

        query = st.text_input(
            "Query",
            value=st.session_state.get("active_query", ""),
            placeholder="e.g.  What is the main contribution of this paper?",
            label_visibility="collapsed",
            key="query_input",
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            ask_clicked = st.button("Run Query →", use_container_width=True)
        with c2:
            top_k_inline = st.number_input(
                "Top-K", min_value=1, max_value=10,
                value=top_k_sb, key="top_k_inline"
            )

        # example queries — clicking one sets session state and re-runs
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                    text-transform:uppercase;color:var(--muted);margin-top:1rem;margin-bottom:0.45rem;">
            Example queries
        </div>""", unsafe_allow_html=True)

        eg_cols = st.columns(2)
        examples = [
            "Summarise key findings",
            "What methodology was used?",
            "List all conclusions",
            "Explain the architecture",
        ]
        for i, ex in enumerate(examples):
            with eg_cols[i % 2]:
                if st.button(ex, key=f"eg_{i}", use_container_width=True):
                    st.session_state["active_query"] = ex
                    st.rerun()


# ═══════════════════════════════ RIGHT ════════════════════════════════════════
with col_right:
    st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:var(--subtext);margin-bottom:0.75rem;">💡  Answer & Sources</div>', unsafe_allow_html=True)

    run = is_ready and ask_clicked and bool(query and query.strip())

    if run:
        st.session_state["active_query"] = query   # persist for next render

        with st.spinner("Retrieving relevant chunks…"):
            result = st.session_state["pipeline"].ask(query, k=int(top_k_inline))

        sources  = result["sources"]
        thresh   = st.session_state.get("score_thresh", 0.0)
        filtered = [s for s in sources if s["score"] >= thresh] or sources[:1]
        top_sc   = max(s["score"] for s in filtered)

        st.session_state["last_n_results"] = len(filtered)
        st.session_state["last_top_score"] = f"{top_sc:.3f}"

        # store last result for persistent display
        st.session_state["last_result"]   = result
        st.session_state["last_filtered"] = filtered
        st.session_state["last_query_display"] = query

    # ── Display last result (persists across reruns) ──
    if st.session_state.get("last_result") and is_ready:
        result   = st.session_state["last_result"]
        filtered = st.session_state["last_filtered"]
        q_disp   = st.session_state.get("last_query_display", "")
        q_preview = (q_disp[:60] + "…") if len(q_disp) > 60 else q_disp

        st.markdown(f"""
        <div class="answer-wrap">
            <div class="answer-header">
                <div class="answer-header-dot"></div>
                <div class="answer-header-label">Answer</div>
                <div class="answer-header-query">"{q_preview}"</div>
            </div>
            <div class="answer-body">{result["answer"]}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sources-header">Retrieved Sources · {len(filtered)} chunks</div>', unsafe_allow_html=True)

        for src in filtered:
            snippet = src["text"][:300].replace("<", "&lt;").replace(">", "&gt;")
            if len(src["text"]) > 300:
                snippet += "…"
            bar_w = min(int(src["score"] * 100), 100)
            st.markdown(f"""
            <div class="source-card">
                <div class="source-meta">
                    <span class="source-id">chunk #{src['chunk_id']}</span>
                    <span class="source-score">score {src['score']:.4f}</span>
                    <div style="margin-left:auto;width:80px;height:4px;background:var(--border2);border-radius:99px;overflow:hidden;">
                        <div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,var(--accent),var(--teal));border-radius:99px;"></div>
                    </div>
                </div>
                <div class="source-text">{snippet}</div>
            </div>""", unsafe_allow_html=True)

        with st.expander("View raw result dict"):
            st.json(result)

    elif is_ready:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">Ready to query</div>
            <div class="empty-sub">Type a question on the left and click <strong>Run Query →</strong></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <div class="empty-title">Waiting for a document</div>
            <div class="empty-sub">Upload a PDF in the sidebar to build the index.</div>
        </div>""", unsafe_allow_html=True)

    # ── Query history ──
    if run and query.strip():
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].insert(0, {"q": query, "n": len(filtered)})
        st.session_state["history"] = st.session_state["history"][:8]

    if st.session_state.get("history") and is_ready:
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("Query History"):
            for i, item in enumerate(st.session_state["history"]):
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:6px 0;
                            border-bottom:1px solid var(--border);font-size:0.82rem;">
                    <span style="font-family:'DM Mono',monospace;color:var(--muted);font-size:0.7rem;">#{i+1}</span>
                    <span style="color:var(--text);flex:1;">{item['q']}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--subtext);">{item['n']} src</span>
                </div>""", unsafe_allow_html=True)