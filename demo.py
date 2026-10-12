import html
import os
import re

import gradio as gr
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
MODEL_PATH = "simcse-model"   # the model trained by train.py
DOC_FILE = "demo.txt"         # one document per line; replace this file for a new demo
TOP_K = 3                     # how many results to show
# 70% our model + 30% BM25 (the mix found in hybrid.py)
ALPHA = 0.7

# ---------------------------------------------------------------
# Model and documents
# ---------------------------------------------------------------
print("Loading model ...")
model = SentenceTransformer(MODEL_PATH)

_state = {"mtime": None, "docs": [], "emb": None, "bm25": None}


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def load_docs():
    """Read DOC_FILE. If the file was changed since the last search, reload it."""
    if not os.path.exists(DOC_FILE):
        return False
    mtime = os.path.getmtime(DOC_FILE)
    if mtime != _state["mtime"]:
        with open(DOC_FILE, encoding="utf-8") as f:
            docs = [line.strip() for line in f if line.strip()]
        if docs:
            _state["docs"] = docs
            _state["emb"] = model.encode(docs, normalize_embeddings=True,
                                         show_progress_bar=False)
            _state["bm25"] = BM25Okapi([tokenize(d) for d in docs])
            print(f"Loaded {len(docs)} documents from {DOC_FILE}")
        else:
            _state["docs"] = []
        _state["mtime"] = mtime
    return len(_state["docs"]) > 0


load_docs()

# ---------------------------------------------------------------
# Design: midnight blue page, emerald/cyan accent.
# Colors are fixed (!important), so it looks the same in light or dark mode.
# To change the accent color, edit --ss-accent and --ss-accent2 below.
# ---------------------------------------------------------------
STYLE = """
<style>
  :root {
    --ss-bg: #0b1220;
    --ss-panel: #101a30;
    --ss-border: #1e2d4a;
    --ss-text: #e2e8f0;
    --ss-muted: #94a3b8;
    --ss-accent: #10b981;
    --ss-accent2: #06b6d4;
  }

  /* page */
  body, gradio-app, .gradio-container, .main, .app {
    background:
      radial-gradient(1000px 420px at 50% -120px, rgba(6,182,212,.18), transparent 70%),
      var(--ss-bg) !important;
  }
  .gradio-container { max-width: 820px !important; margin: 0 auto !important; }
  .gradio-container .block, .gradio-container .form, .gradio-container .row {
    background: transparent !important; border: none !important; box-shadow: none !important;
  }
  footer { display: none !important; }

  /* search box */
  .gradio-container textarea, .gradio-container input[type="text"] {
    background: var(--ss-panel) !important;
    color: var(--ss-text) !important;
    border: 1px solid var(--ss-border) !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    padding: 14px 16px !important;
    box-shadow: none !important;
  }
  .gradio-container textarea::placeholder { color: #64748b !important; }
  .gradio-container textarea:focus {
    border-color: var(--ss-accent) !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,.22) !important;
    outline: none !important;
  }

  /* search button */
  .gradio-container button.primary {
    background: linear-gradient(135deg, var(--ss-accent), var(--ss-accent2)) !important;
    color: #04231b !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 14px !important;
    min-height: 52px !important;
    box-shadow: 0 6px 18px rgba(16,185,129,.25) !important;
  }
  .gradio-container button.primary:hover { filter: brightness(1.08); }

  /* header */
  .ss-head { text-align: center; padding: 26px 0 10px; }
  .ss-logo { width: 54px; height: 54px; margin: 0 auto 12px; border-radius: 16px;
             display: flex; align-items: center; justify-content: center;
             background: linear-gradient(135deg, var(--ss-accent), var(--ss-accent2));
             box-shadow: 0 10px 28px rgba(6,182,212,.30); }
  .ss-title { margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -.5px;
              color: var(--ss-text) !important; }
  .ss-sub { margin: 6px 0 0; font-size: 14px; color: var(--ss-muted) !important; }

  /* results */
  .ss-empty { text-align: center; padding: 34px 12px; font-size: 14px; color: #64748b !important; }
  .ss-count { font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
              color: #64748b !important; margin: 6px 4px 12px; }

  .ss-card { background: var(--ss-panel) !important; border: 1px solid var(--ss-border);
             border-radius: 16px; padding: 16px 18px; margin-bottom: 12px;
             animation: ss-in .35s ease both; }
  .ss-card.best { border-color: var(--ss-accent);
                  background: linear-gradient(180deg, #0e2a2b, var(--ss-panel)) !important;
                  box-shadow: 0 0 0 1px rgba(16,185,129,.30), 0 12px 32px rgba(16,185,129,.10); }
  .ss-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .ss-rank { width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center;
             justify-content: center; font-size: 13px; font-weight: 700;
             background: #1e2d4a !important; color: #93c5fd !important; }
  .ss-card.best .ss-rank { background: var(--ss-accent) !important; color: #04231b !important; }
  .ss-tag { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
            color: var(--ss-accent) !important; }
  .ss-text { font-size: 16px; line-height: 1.65; color: var(--ss-text) !important; }
  .ss-bar { height: 4px; border-radius: 4px; background: #1e2d4a; margin-top: 14px; overflow: hidden; }
  .ss-fill { height: 4px; border-radius: 4px;
             background: linear-gradient(90deg, var(--ss-accent), var(--ss-accent2)); }

  .ss-note { background: #2a2110 !important; color: #fcd34d !important; border: 1px solid #5b4514;
             border-radius: 14px; padding: 14px 16px; font-size: 14px; }

  @keyframes ss-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
</style>
"""

HEADER = STYLE + """
<div class="ss-head">
  <div class="ss-logo">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#04231b"
         stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="7"></circle><line x1="21" y1="21" x2="16.5" y2="16.5"></line>
    </svg>
  </div>
  <h1 class="ss-title">Semantic Search</h1>
  <p class="ss-sub">Ask a question. Results are matched by meaning, not just exact words.</p>
</div>
"""

EMPTY = '<div class="ss-empty">Results will appear here.</div>'


def note(message):
    return f'<div class="ss-note">{html.escape(message)}</div>'


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-9)


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------
def search(query):
    query = (query or "").strip()
    if not query:
        return EMPTY
    if not load_docs():
        return note(f"Could not find any text in '{DOC_FILE}'. Put it in the same folder "
                    "as demo_app.py, with one document per line.")

    docs = _state["docs"]
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    meaning = _state["emb"] @ q_emb
    keywords = np.array(_state["bm25"].get_scores(tokenize(query)))
    score = ALPHA * zscore(meaning) + (1 - ALPHA) * zscore(keywords)

    order = np.argsort(-score)[:TOP_K]
    lo, hi = float(score.min()), float(score.max())

    cards = f'<div class="ss-count">Top {len(order)} of {len(docs)} documents</div>'
    for rank, i in enumerate(order, start=1):
        pct = int(round((float(score[i]) - lo) / (hi - lo + 1e-9) * 100))
        best = rank == 1
        tag = '<span class="ss-tag">Best match</span>' if best else ""
        delay = (rank - 1) * 0.07
        cards += f"""
        <div class="ss-card {'best' if best else ''}" style="animation-delay:{delay:.2f}s">
          <div class="ss-row"><span class="ss-rank">{rank}</span>{tag}</div>
          <div class="ss-text">{html.escape(docs[i])}</div>
          <div class="ss-bar"><div class="ss-fill" style="width:{pct}%"></div></div>
        </div>"""
    return cards


# ---------------------------------------------------------------
# Page: just a search box and the results
# ---------------------------------------------------------------
with gr.Blocks(title="Semantic Search") as app:
    gr.HTML(HEADER)
    with gr.Row():
        query = gr.Textbox(
            show_label=False,
            container=False,
            placeholder="Type your question and press Enter ...",
            lines=1,
            scale=5,
        )
        search_btn = gr.Button(
            "Search", variant="primary", scale=1, min_width=110)
    results = gr.HTML(value=EMPTY)

    search_btn.click(search, query, results)
    query.submit(search, query, results)

if __name__ == "__main__":
    app.launch()
