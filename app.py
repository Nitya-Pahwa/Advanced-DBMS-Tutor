"""
App: Streamlit UI for DBMS Concept Graph Tutor

Features:
- Chat-based DBMS Q&A
- RAG pipeline integration
- Concept graph visualization
- Multi-session chat history
- Voice assistant (in-browser mic recording via streamlit-audio-recorder)
- MCQ Quiz generator
"""

import streamlit as st
from graph_flow import compile_graph
from Nodes.safe_llm import safe_invoke
from voice_handler import transcribe_audio, text_to_speech
from Nodes.quiz_generator import generate_quiz
from pyvis.network import Network
from audio_recorder_streamlit import audio_recorder
import re, json, tempfile, os


st.set_page_config(
    page_title="DBMS Concept Tutor",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----- CSS ---------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0d0f18;
    --surface: #13161f;
    --card:    #1a1e2e;
    --accent:  #00e5ff;
    --purple:  #7c4dff;
    --green:   #00e676;
    --yellow:  #ffab00;
    --red:     #ff5252;
    --text:    #e8eaf6;
    --muted:   #7986cb;
    --border:  #222540;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header,
[data-testid="collapsedControl"],
[data-testid="stSidebar"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* ── LEFT PANEL ── */
section.main > div.block-container > div > div > div > div:nth-child(1) {
    background: var(--surface) !important;
}
[data-testid="column"]:first-child {
    background: var(--surface);
    border-right: 1px solid var(--border);
    min-height: 100vh;
    padding: 1rem 0.8rem 2rem 0.8rem !important;
}
[data-testid="column"]:first-child > div { padding: 0 !important; }

/* ── RIGHT PANEL ── */
[data-testid="column"]:last-child {
    padding: 1.2rem 2rem 2rem 2rem !important;
}
[data-testid="column"]:last-child > div { padding: 0 !important; }

/* ── Topbar ── */
.topbar {
    display: flex;
    align-items: center;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}
.topbar-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.3px;
}
.topbar-title span { color: var(--purple); }

/* ── Panel title ── */
.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.7rem;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.1) !important;
}

/* ── ALL buttons base ── */
.stButton > button {
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    transition: filter 0.15s !important;
}
.stButton > button:hover { filter: brightness(1.18) !important; }

/* Ask button */
.ask-btn .stButton > button {
    background: linear-gradient(135deg, #7c4dff, #4527a0) !important;
    color: white !important;
    width: 100% !important;
    padding: 0.65rem 1rem !important;
}

/* New Chat button */
.new-chat-btn .stButton > button {
    background: linear-gradient(135deg, #00897b, #00695c) !important;
    color: white !important;
    width: 100% !important;
    padding: 0.42rem 0.8rem !important;
    font-size: 0.78rem !important;
    margin-bottom: 0.6rem !important;
}

/* Session list buttons */
.sess-btn .stButton > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.38rem 0.65rem !important;
    width: 100% !important;
    margin-bottom: 0 !important;
}
.sess-btn-active .stButton > button {
    background: rgba(0,229,255,0.08) !important;
    color: var(--accent) !important;
    border-color: rgba(0,229,255,0.28) !important;
    font-weight: 600 !important;
}

/* ── Chat bubbles ── */
.bubble-q {
    background: rgba(124,77,255,0.1);
    border: 1px solid rgba(124,77,255,0.2);
    border-radius: 10px 10px 10px 2px;
    padding: 0.65rem 1rem;
    margin: 1rem 0 0.3rem 0;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text);
}
.bubble-a {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 2px 10px 10px 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: var(--text);
}

/* ── Badges ── */
.tbadge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    padding: 0.13rem 0.48rem;
    border-radius: 4px;
    font-weight: 700;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 0.4rem;
}
.t-definition   { background:rgba(0,230,118,.12); color:#00e676; border:1px solid rgba(0,230,118,.25);}
.t-comparison   { background:rgba(255,171,0,.12);  color:#ffab00; border:1px solid rgba(255,171,0,.25);}
.t-relationship { background:rgba(124,77,255,.12); color:#b388ff; border:1px solid rgba(124,77,255,.25);}
.t-process      { background:rgba(0,229,255,.12);  color:#00e5ff; border:1px solid rgba(0,229,255,.25);}
.t-sql          { background:rgba(255,82,82,.12);  color:#ff5252; border:1px solid rgba(255,82,82,.25);}

/* ── Source pills ── */
.pill {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.67rem;
    font-family: 'Space Mono', monospace;
    padding: 0.13rem 0.48rem;
    border-radius: 4px;
    margin: 0.15rem 0.15rem 0 0;
}
.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0.7rem 0 0.3rem 0;
}
hr.div { border:none; border-top:1px solid var(--border); margin:1rem 0; }

/* ── Empty state ── */
.empty {
    text-align: center;
    padding: 4rem 1rem 2rem 1rem;
    color: var(--muted);
}
.empty .big  { font-family:'Space Mono',monospace; font-size:0.88rem; margin-bottom:0.7rem; }
.empty .small{ font-size:0.73rem; opacity:0.55; line-height:1.6; }

/* ── Sub-label under session button ── */
.sess-sub {
    font-size: 0.63rem;
    font-family: 'Space Mono', monospace;
    padding-left: 0.5rem;
    margin-bottom: 0.5rem;
    margin-top: 0.1rem;
    opacity: 0.65;
}

/* ── Voice tab ── */
.voice-instruction {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--purple);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.8;
    margin-bottom: 1rem;
}

/* ── Quiz card ── */
.quiz-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.quiz-score {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    padding: 0.6rem 0;
    margin-top: 0.4rem;
}

            /* ── Style the audio_recorder widget as a proper button ── */
.audio-recorder {
    background: linear-gradient(135deg, #7c4dff, #4527a0) !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    cursor: pointer !important;
    border: none !important;
    box-shadow: none !important;
    width: auto !important;
}
.audio-recorder-status {
    display: none !important;
}
.audio-recorder p {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    color: white !important;
    margin: 0 !important;
    letter-spacing: 0.3px !important;
}
.audio-recorder svg {
    color: white !important;
    fill: white !important;
    width: 16px !important;
    height: 16px !important;
}
/* Red pulsing state while recording */
.audio-recorder.recording {
    background: linear-gradient(135deg, #ff5252, #b71c1c) !important;
    animation: pulse-rec 1.2s infinite !important;
}
@keyframes pulse-rec {
    0%   { box-shadow: 0 0 0 0 rgba(255,82,82,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(255,82,82,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,82,82,0); }
}
            
</style>
""", unsafe_allow_html=True)

# ------- Constants ------------------------------------------------------------------------
GRAPH_TYPES = {"definition", "comparison", "relationship"}

# -------- Helpers ----------------------------------------------------------------------------
def clean(answer):
    t = answer.replace("**Answer:**", "").strip()
    if "**Sources used:**" in t:
        t = t.split("**Sources used:**")[0].strip()
    return t

def tbadge(qtype):
    return f'<span class="tbadge t-{qtype}">{qtype}</span>'

def pills(sources):
    return "".join(f'<span class="pill">📎 {s}</span>' for s in sources)

def extract_edges(query, answer):
    prompt = f"""You are a DBMS knowledge graph extractor.
Extract 5-8 concept relationships from the answer.
Return ONLY a JSON array with keys "from", "label", "to".
- "from"/"to": short DBMS concept names (1-4 words)
- "label": short phrase like "is a type of", "depends on", "ensures", "uses"

Question: {query}
Answer: {answer[:700]}

Return only the JSON array, nothing else."""
    raw = safe_invoke(prompt)
    try:
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return [(e["from"], e["label"], e["to"]) for e in data
                    if all(k in e for k in ("from", "label", "to"))]
    except Exception:
        pass
    return []

def build_graph(edges):
    net = Network(height="380px", width="100%", bgcolor="#1e2235",
                  font_color="#e8eaf6", directed=True)
    net.set_options("""{
      "nodes":{"shape":"dot","font":{"size":12,"face":"IBM Plex Sans","color":"#e8eaf6"},"borderWidth":2},
      "edges":{"font":{"size":9,"color":"#9fa8da","align":"middle"},
               "color":{"color":"#7c4dff","highlight":"#00e5ff"},
               "width":1.6,"arrows":{"to":{"enabled":true,"scaleFactor":0.6}},
               "smooth":{"type":"curvedCW","roundness":0.2}},
      "physics":{"barnesHut":{"gravitationalConstant":-8000,"springLength":130},"stabilization":{"iterations":100}}
    }""")
    palette = ["#00e5ff", "#7c4dff", "#00e676", "#ffab00", "#ff5252", "#b388ff", "#80cbc4"]
    seen = {}
    for src, lbl, tgt in edges:
        for name in (src, tgt):
            if name not in seen:
                i = len(seen)
                seen[name] = i
                net.add_node(name, label=name,
                             color=palette[i % len(palette)],
                             size=20 if i == 0 else 14)
        net.add_edge(src, tgt, title=lbl, label=lbl)
    return net.generate_html()

def run_voice_pipeline(voice_query: str):
    """Run RAG pipeline on a transcribed voice query and return all parts."""
    result  = st.session_state.pipeline.invoke(
        {"query": voice_query}, config={"recursion_limit": 50}
    )
    answer  = result.get("final_answer", "No answer.")
    qtype   = result.get("question_type", "definition")
    docs    = result.get("retrieved_docs", [])
    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    text    = clean(answer)
    edges   = extract_edges(voice_query, text) if qtype in GRAPH_TYPES else []
    return answer, qtype, docs, sources, text, edges

# ------ Session state ----------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = compile_graph()
if "sessions" not in st.session_state:
    st.session_state.sessions = [{"id": 0, "name": "Session 1", "messages": []}]
if "active_session" not in st.session_state:
    st.session_state.active_session = 0
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Quiz state
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""

# Voice state
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "voice_result" not in st.session_state:
    st.session_state.voice_result = None


def get_session():
    return st.session_state.sessions[st.session_state.active_session]

def new_session():
    n = len(st.session_state.sessions) + 1
    st.session_state.sessions.append({"id": n - 1, "name": f"Session {n}", "messages": []})
    st.session_state.active_session = len(st.session_state.sessions) - 1
    st.session_state.input_key += 1

# ------- Layout -------------------------------------------------------------------------
left, right = st.columns([1, 4], gap="small")

# -----------------------------------
#  LEFT — History panel
# ------------------------------------
with left:
    st.markdown('<div class="panel-title">💬 Chat History</div>', unsafe_allow_html=True)

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", key="btn_new_chat"):
        new_session()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    for i, sess in enumerate(reversed(st.session_state.sessions)):
        real_i = len(st.session_state.sessions) - 1 - i
        is_active = (real_i == st.session_state.active_session)

        msgs = sess["messages"]
        if msgs:
            label = msgs[0]["query"][:34] + ("…" if len(msgs[0]["query"]) > 34 else "")
            count_txt = f" ({len(msgs)})"
        else:
            label = "Untitled"
            count_txt = ""

        css = "sess-btn-active" if is_active else "sess-btn"
        prefix = "▶ " if is_active else ""

        st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
        if st.button(f"{prefix}{label}{count_txt}", key=f"sess_{real_i}"):
            st.session_state.active_session = real_i
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
#  RIGHT — Main area
# -----------------------------
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    # Topbar
    st.markdown(
        '<div class="topbar">'
        '<p class="topbar-title">🗄️ DBMS Concept <span>Graph Tutor</span></p>'
        '</div>',
        unsafe_allow_html=True
    )

    tab_qa, tab_voice, tab_quiz = st.tabs(["💬 Q&A", "🎙️ Voice", "📝 Quiz"])


    # --------------------------------------------------------------------------
    #  TAB 1 — TEXT Q&A
    # --------------------------------------------------------------------------
    with tab_qa:
        in_col, btn_col = st.columns([5, 1])
        with in_col:
            query = st.text_input(
                "", placeholder="Ask a DBMS question…",
                label_visibility="collapsed",
                key=f"q_input_{st.session_state.input_key}"
            )
        with btn_col:
            st.markdown('<div class="ask-btn">', unsafe_allow_html=True)
            asked = st.button("Ask →", key="btn_ask")
            st.markdown('</div>', unsafe_allow_html=True)

        if asked and query.strip():
            with st.spinner("Thinking…"):
                result = st.session_state.pipeline.invoke(
                    {"query": query}, config={"recursion_limit": 50}
                )
            answer  = result.get("final_answer", "No answer.")
            qtype   = result.get("question_type", "definition")
            docs    = result.get("retrieved_docs", [])
            sources = sorted({d.metadata.get("source", "unknown") for d in docs})
            text    = clean(answer)

            edges = []
            if qtype in GRAPH_TYPES:
                with st.spinner("Building concept graph…"):
                    edges = extract_edges(query, text)

            sess_idx = st.session_state.active_session
            st.session_state.sessions[sess_idx]["messages"].append({
                "query": query, "answer": answer, "qtype": qtype,
                "sources": sources, "docs": docs, "edges": edges
            })
            if len(st.session_state.sessions[sess_idx]["messages"]) == 1:
                st.session_state.sessions[sess_idx]["name"] = query[:34]

            st.session_state.input_key += 1
            st.rerun()

        sess     = get_session()
        messages = sess["messages"]

        if not messages:
            st.markdown(
                '<div class="empty">'
                '<div class="big">⬆ Ask your first question for this session</div>'
                '<div class="small">'
                'Try: "What is BCNF?" · "Compare 2NF and 3NF" · "How does indexing work?"'
                '</div></div>',
                unsafe_allow_html=True
            )
        else:
            for msg in messages:
                qtype   = msg["qtype"]
                sources = msg["sources"]
                edges   = msg["edges"]
                text    = clean(msg["answer"])

                st.markdown(
                    f'<div class="bubble-q">'
                    f'<span style="color:var(--muted);font-size:0.72rem;font-family:Space Mono,monospace;">Q</span> '
                    f'{msg["query"]}{tbadge(qtype)}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<div class="bubble-a">{text}</div>', unsafe_allow_html=True)

                src_col, graph_col = st.columns([1, 2])

                with src_col:
                    st.markdown('<div class="sec-label"> Sources</div>', unsafe_allow_html=True)
                    if sources:
                        st.markdown(pills(sources), unsafe_allow_html=True)
                        with st.expander("View snippets"):
                            for d in msg.get("docs", []):
                                snip = d.page_content[:250].replace("\n", " ")
                                src  = d.metadata.get("source", "unknown")
                                st.markdown(f"**{src}**")
                                st.markdown(
                                    f'<div style="font-size:0.78rem;color:#9fa8da;'
                                    f'background:var(--bg);padding:0.4rem 0.6rem;'
                                    f'border-radius:6px;margin-bottom:0.3rem">{snip}…</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.markdown('<span style="color:#7986cb;font-size:0.8rem">No sources.</span>',
                                    unsafe_allow_html=True)

                with graph_col:
                    if qtype in GRAPH_TYPES:
                        if edges:
                            st.markdown(
                                '<div style="font-family:Space Mono,monospace;font-size:0.8rem;'
                                'color:var(--purple);margin-bottom:0.2rem;"> Concept Graph</div>'
                                '<div style="font-size:0.7rem;color:var(--muted);margin-bottom:0.3rem;">'
                                'Nodes = concepts · Arrows = relationships</div>',
                                unsafe_allow_html=True
                            )
                            st.components.v1.html(build_graph(edges), height=400, scrolling=False)
                        else:
                            st.markdown(
                                '<div style="color:var(--muted);font-size:0.8rem;padding-top:0.5rem">'
                                'Could not extract concept relationships.</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            f'<div style="color:var(--muted);font-size:0.8rem;padding-top:0.5rem;line-height:1.8">'
                            f'No graph for <strong style="color:var(--red)">{qtype}</strong> questions.<br>'
                            f'Graph shows for: '
                            f'<span class="tbadge t-definition">definition</span> '
                            f'<span class="tbadge t-comparison">comparison</span> '
                            f'<span class="tbadge t-relationship">relationship</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown('<hr class="div">', unsafe_allow_html=True)


    # --------------------------------------------------------------------------
    #  TAB 2 — VOICE (in-browser mic, no file upload)
    # ---------------------------------------------------------------------------
    with tab_voice:
        from streamlit_mic_recorder import mic_recorder

        st.markdown('<div class="sec-label">🎙️ Voice Assistant</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="voice-instruction">'
            '1. Click <strong style="color:var(--accent)">🎙️ Record Audio</strong> to start.<br>'
            '2. Speak your DBMS question clearly.<br>'
            '3. Click <strong style="color:var(--accent)">⏹️ Stop</strong> when done.<br>'
            '4. Your recorded audio plays back for confirmation.<br>'
            '5. The answer appears as text and is read aloud automatically.'
            '</div>',
            unsafe_allow_html=True
        )

        audio = mic_recorder(
            start_prompt="🎙️  Record Audio",
            stop_prompt="⏹️  Stop Recording",
            just_once=True,
            use_container_width=False,
            key="mic_recorder"
        )

        if audio and audio["bytes"] != st.session_state.last_audio_bytes:
            audio_bytes = audio["bytes"]
            st.session_state.last_audio_bytes = audio_bytes
            st.session_state.voice_result = None


            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                with st.spinner("Transcribing…"):
                    voice_query = transcribe_audio(tmp_path)

                if not voice_query.strip():
                    st.warning("Could not hear anything clearly. Please try again.")
                else:
                    with st.spinner("Thinking…"):
                        answer, qtype, docs, sources, text, edges = run_voice_pipeline(voice_query)

                    with st.spinner("Generating audio response…"):
                        tts_bytes = text_to_speech(text)

                    st.session_state.voice_result = {
                        "query":      voice_query,
                        "text":       text,
                        "answer":     answer,
                        "tts_bytes":  tts_bytes,
                        "user_audio": audio_bytes,
                        "qtype":      qtype,
                        "sources":    sources,
                        "docs":       docs,
                        "edges":      edges,
                    }

                    sess_idx = st.session_state.active_session
                    st.session_state.sessions[sess_idx]["messages"].append({
                        "query":   f"🎙️ {voice_query}",
                        "answer":  answer,
                        "qtype":   qtype,
                        "sources": sources,
                        "docs":    docs,
                        "edges":   edges,
                    })
            finally:
                os.unlink(tmp_path)

        vr = st.session_state.voice_result
        if vr:
            st.markdown('<div class="sec-label">🎧 Your Recording</div>', unsafe_allow_html=True)
            st.audio(vr["user_audio"], format="audio/wav")
            st.markdown(f'<div class="bubble-q">🎙️ <em>{vr["query"]}</em></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble-a">{vr["text"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">🔊 Audio Answer</div>', unsafe_allow_html=True)
            st.audio(vr["tts_bytes"], format="audio/mp3")
            if vr["sources"]:
                st.markdown('<div class="sec-label">Sources</div>', unsafe_allow_html=True)
                st.markdown(pills(vr["sources"]), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty" style="padding:2rem 1rem">'
                '<div class="big">🎙️ Press Record Audio and ask a DBMS question</div>'
                '<div class="small">Your recording and the spoken answer will both appear here.</div>'
                '</div>',
                unsafe_allow_html=True
            )

    # ----------------------------------------------------------------------------
    #  TAB 3 — MCQ QUIZ
    # ------------------------------------------------------------------------------
    with tab_quiz:
        st.markdown('<div class="sec-label">📝 DBMS MCQ Quiz</div>', unsafe_allow_html=True)

        qz_col1, qz_col2, qz_col3 = st.columns([3, 1, 1])
        with qz_col1:
            quiz_topic = st.text_input(
                "Topic",
                placeholder="e.g. SQL, Normalization, Indexing, Transactions…",
                key="quiz_topic_input"
            )
        with qz_col2:
            num_q = st.selectbox("Questions", [5, 10], key="quiz_num")
        with qz_col3:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            gen_btn = st.button("Generate Quiz", key="btn_gen_quiz")

        if gen_btn and quiz_topic.strip():
            with st.spinner(f"Generating {num_q} questions on '{quiz_topic}'…"):
                qs = generate_quiz(quiz_topic.strip(), num_q)
            if qs:
                st.session_state.quiz_questions = qs
                st.session_state.quiz_answers   = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_topic     = quiz_topic.strip()
                st.rerun()
            else:
                st.error(
                    f"Could not generate a quiz for '{quiz_topic}'. "
                    "Check the terminal for the raw LLM response and try again."
                )

        qs = st.session_state.quiz_questions
        if qs:
            st.markdown(
                f'<div style="font-size:0.8rem;color:var(--muted);margin-bottom:0.8rem">'
                f'Topic: <strong style="color:var(--accent)">{st.session_state.quiz_topic}</strong>'
                f' · {len(qs)} questions</div>',
                unsafe_allow_html=True
            )

            for i, q in enumerate(qs):
                st.markdown(
                    f'<div class="quiz-card">'
                    f'<strong style="color:var(--text)">Q{i+1}. {q["question"]}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Extract letter from "A. some text" → "A"
                option_letters = [opt.split(".")[0].strip() for opt in q["options"]]

                chosen = st.radio(
                    label=f"q_{i}",
                    options=option_letters,
                    format_func=lambda x, _q=q: next(
                        o for o in _q["options"] if o.startswith(x)
                    ),
                    index=None,
                    key=f"quiz_radio_{i}",
                    disabled=st.session_state.quiz_submitted,
                    label_visibility="collapsed"
                )

                if chosen:
                    st.session_state.quiz_answers[i] = chosen

                if st.session_state.quiz_submitted:
                    correct = q["answer"]
                    user    = st.session_state.quiz_answers.get(i, "")
                    if user == correct:
                        st.markdown(
                            '<span style="color:#00e676;font-size:0.8rem">✅ Correct!</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<span style="color:#ff5252;font-size:0.8rem">❌ Wrong. '
                            f'Correct answer: <strong>{correct}</strong></span>',
                            unsafe_allow_html=True
                        )

            st.markdown("---")

            if not st.session_state.quiz_submitted:
                if st.button("Submit Quiz", key="btn_submit_quiz"):
                    st.session_state.quiz_submitted = True
                    st.rerun()
            else:
                score  = sum(
                    1 for i, q in enumerate(qs)
                    if st.session_state.quiz_answers.get(i) == q["answer"]
                )
                pct    = int(score / len(qs) * 100)
                colour = "#00e676" if pct >= 70 else "#ffab00" if pct >= 40 else "#ff5252"

                st.markdown(
                    f'<div class="quiz-score" style="color:{colour}">'
                    f'Score: {score}/{len(qs)} ({pct}%)'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if st.button("🔄 Retake / New Quiz", key="btn_retake"):
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_submitted  = False
                    st.session_state.quiz_answers    = {}
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
