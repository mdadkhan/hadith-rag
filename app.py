"""
app.py
------
Simple Flask web server that exposes query.py through an HTML UI.

Models are loaded ONCE at startup — no reload penalty per request.

Usage:
    python3 app.py
    then open http://localhost:5000 in your browser

Install Flask once:
    pip3 install flask
"""

import os
import sys

from flask import Flask, jsonify, render_template_string, request

# Ensure the project directory is on the path so config/query imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import FINAL_TOP_K, RETRIEVAL_K, get_collection, load_embed_model, load_rerank_model, EMBED_MODEL, RERANK_MODEL
from query import generate_answer, hybrid_retrieve

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------

print(f"Loading embedding model '{EMBED_MODEL}'...")
_embed_model = load_embed_model(EMBED_MODEL)

print(f"Loading re-ranking model '{RERANK_MODEL}'...")
_rerank_model = load_rerank_model(RERANK_MODEL)

_, _collection = get_collection()

if _collection.count() == 0:
    sys.exit("ERROR: Vector DB is empty. Run an ingest script first.")

print(f"\nModels ready. Vector DB has {_collection.count()} documents.")
print("Open http://localhost:8080 in your browser.\n")

# ---------------------------------------------------------------------------
# HTML template (single-file, no external dependencies)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Hadith RAG Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Amiri&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f0;
    color: #1a1a1a;
    min-height: 100vh;
    padding: 2rem 1rem;
  }

  .container { max-width: 860px; margin: 0 auto; }

  h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: #1a3a2a;
  }
  .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 1.75rem; }

  .search-box {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }
  textarea {
    flex: 1;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    font-family: inherit;
    border: 1.5px solid #ccc;
    border-radius: 8px;
    resize: vertical;
    min-height: 72px;
    transition: border-color 0.2s;
    background: #fff;
  }
  textarea:focus { outline: none; border-color: #2d7a4f; }

  button {
    padding: 0 1.5rem;
    background: #2d7a4f;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s, opacity 0.2s;
    white-space: nowrap;
    align-self: flex-start;
    height: 72px;
  }
  button:hover:not(:disabled) { background: #245f3e; }
  button:disabled { opacity: 0.55; cursor: not-allowed; }

  .spinner {
    display: none;
    align-items: center;
    gap: 0.5rem;
    color: #555;
    font-size: 0.9rem;
    margin-bottom: 1rem;
  }
  .spinner.active { display: flex; }
  .spinner-ring {
    width: 18px; height: 18px;
    border: 2px solid #ccc;
    border-top-color: #2d7a4f;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .answer-card {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    line-height: 1.65;
    display: none;
  }
  .answer-card h2 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #2d7a4f;
    margin-bottom: 0.75rem;
  }
  .answer-text { white-space: pre-wrap; }

  .sources-section { display: none; }
  .sources-section h2 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    margin-bottom: 0.75rem;
  }
  .source-card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-left: 3px solid #2d7a4f;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
  }
  .source-meta {
    font-size: 0.78rem;
    font-weight: 600;
    color: #2d7a4f;
    margin-bottom: 0.35rem;
  }
  .source-preview {
    font-size: 0.85rem;
    color: #444;
    line-height: 1.7;
    white-space: pre-wrap;
    unicode-bidi: plaintext;   /* let browser detect RTL/LTR per paragraph */
  }
  /* Arabic runs inside the preview get their own font and direction */
  .source-preview:lang(ar),
  .arabic {
    font-family: 'Amiri', 'Scheherazade New', 'Traditional Arabic', serif;
    direction: rtl;
    text-align: right;
  }

  .error {
    background: #fff0f0;
    border: 1px solid #f5c0c0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #c0392b;
    display: none;
    margin-bottom: 1rem;
  }
</style>
</head>
<body>
<div class="container">
  <h1>&#x1F4D6; Hadith Search</h1>
  <p class="subtitle">Ask a question — Riyadh al-Saliheen, Sahih Bukhari &amp; Sahih Muslim</p>

  <div class="search-box">
    <textarea id="question" placeholder="e.g. What is the reward of charity during Ramadan?" rows="3"></textarea>
    <button id="askBtn" onclick="askQuestion()">Ask</button>
  </div>

  <div class="spinner" id="spinner">
    <div class="spinner-ring"></div>
    <span>Searching &amp; generating answer…</span>
  </div>

  <div class="error" id="errorBox"></div>

  <div class="answer-card" id="answerCard">
    <h2>Answer</h2>
    <div class="answer-text" id="answerText"></div>
  </div>

  <div class="sources-section" id="sourcesSection">
    <h2>Sources used</h2>
    <div id="sourcesList"></div>
  </div>
</div>

<script>
  const questionEl  = document.getElementById("question");
  const askBtn      = document.getElementById("askBtn");
  const spinner     = document.getElementById("spinner");
  const errorBox    = document.getElementById("errorBox");
  const answerCard  = document.getElementById("answerCard");
  const answerText  = document.getElementById("answerText");
  const sourcesSec  = document.getElementById("sourcesSection");
  const sourcesList = document.getElementById("sourcesList");

  questionEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askQuestion(); }
  });

  async function askQuestion() {
    const question = questionEl.value.trim();
    if (!question) return;

    // Reset UI
    askBtn.disabled = true;
    spinner.classList.add("active");
    errorBox.style.display = "none";
    answerCard.style.display = "none";
    sourcesSec.style.display = "none";
    sourcesList.innerHTML = "";

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error || res.statusText);
      }

      const data = await res.json();

      // Show answer
      answerText.textContent = data.answer;
      answerCard.style.display = "block";

      // Show sources
      if (data.chunks && data.chunks.length > 0) {
        data.chunks.forEach((c, i) => {
          const ref     = c.reference ? ` &nbsp;|&nbsp; ${c.reference}` : "";
          const meta    = `[${i+1}] ${c.source}${ref} &nbsp;|&nbsp; Page ${c.page}`;
          const preview = c.preview.length > 300 ? c.preview.slice(0, 300) + "…" : c.preview;
          sourcesList.insertAdjacentHTML("beforeend", `
            <div class="source-card">
              <div class="source-meta">${meta}</div>
              <div class="source-preview">${escHtml(preview)}</div>
            </div>`);
        });
        sourcesSec.style.display = "block";
      }
    } catch (err) {
      errorBox.textContent = "Error: " + err.message;
      errorBox.style.display = "block";
    } finally {
      spinner.classList.remove("active");
      askBtn.disabled = false;
    }
  }

  function escHtml(s) {
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/query", methods=["POST"])
def query():
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        chunks = hybrid_retrieve(
            question, _collection, _embed_model, _rerank_model,
            retrieval_k=RETRIEVAL_K, final_top_k=FINAL_TOP_K,
        )
    except Exception as e:
        return jsonify({"error": f"Retrieval error: {e}"}), 500

    if not chunks:
        return jsonify({"answer": "No relevant content found in the knowledge base.", "chunks": []})

    try:
        answer = generate_answer(question, chunks)
    except Exception as e:
        return jsonify({"error": f"LLM error: {e}"}), 500

    serialised_chunks = [
        {
            "source"   : c["source"],
            "page"     : c["page"],
            "reference": c.get("reference", ""),
            "preview"  : c["text"][:400],
        }
        for c in chunks
    ]
    return jsonify({"answer": answer, "chunks": serialised_chunks})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
