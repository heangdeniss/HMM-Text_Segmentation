/* UI wiring for the HMM POS Tagger demo. */
(function () {
  "use strict";

  const { createEngine, tokenize, TAG_META } = window.HMMTagger;
  const MODEL_URL = "model/hmm_pos_model.json";
  const MAX_TOKENS = 600;

  const EXAMPLES = [
    {
      label: "Quick fox",
      text: "The quick brown fox jumps over the lazy dog .",
    },
    {
      label: "Noun / verb ambiguity",
      text: "It was a very tempting offer .",
    },
    {
      label: "Frequency bias",
      text: "With tips , the girls average between $150 and $200 a week , depending on basic salary .",
    },
    {
      label: "Long sentence",
      text: "According to state law a slave had to be at least thirty years old before he could be freed .",
    },
    {
      label: "Unseen words",
      text: "The blockchain startup open-sourced its transformer pipeline in Phnom Penh .",
    },
  ];

  const els = {
    input: document.getElementById("inputText"),
    tag: document.getElementById("tagBtn"),
    clear: document.getElementById("clearBtn"),
    output: document.getElementById("output"),
    timing: document.getElementById("timing"),
    status: document.getElementById("status"),
    pill: document.getElementById("statusPill"),
    pillText: document.getElementById("statusPillText"),
    examples: document.getElementById("examples"),
    legend: document.getElementById("tagLegend"),
  };

  let engine = null;
  let readyStatus = "";

  function setPill(state, text) {
    els.pill.dataset.state = state;
    els.pillText.textContent = text;
  }

  function setStatus(text, isError) {
    els.status.textContent = text;
    els.status.classList.toggle("error", Boolean(isError));
  }

  function renderLegend() {
    const order = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", ".", "X"];
    const fragment = document.createDocumentFragment();

    for (const tag of order) {
      const meta = TAG_META[tag];
      if (!meta) continue;

      const item = document.createElement("li");
      item.className = "tag-item";
      item.style.setProperty("--tag-color", meta.color);

      const dot = document.createElement("span");
      dot.className = "tag-dot";
      dot.setAttribute("aria-hidden", "true");

      const body = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = meta.short || tag;
      const description = document.createElement("span");
      description.textContent = `${meta.label} — ${meta.description}`;

      body.append(name, description);
      item.append(dot, body);
      fragment.append(item);
    }

    els.legend.append(fragment);
  }

  function renderExamples() {
    for (const example of EXAMPLES) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example";
      button.textContent = example.label;
      button.addEventListener("click", () => {
        els.input.value = example.text;
        tagInput();
      });
      els.examples.append(button);
    }
  }

  function clearOutput(message) {
    els.output.replaceChildren();
    const placeholder = document.createElement("p");
    placeholder.className = "placeholder";
    placeholder.textContent = message;
    els.output.append(placeholder);
    els.timing.textContent = "";
  }

  function render(results, elapsedMs) {
    if (results.length === 0) {
      clearOutput("Type a sentence to tag it.");
      return;
    }

    const fragment = document.createDocumentFragment();
    let tokenCount = 0;

    for (const result of results) {
      const article = document.createElement("article");
      article.className = "sentence-result";

      const tokens = document.createElement("div");
      tokens.className = "tokens";

      result.tokens.forEach((word, index) => {
        const tag = result.tags[index];
        const meta = TAG_META[tag] || { label: tag, color: "#94a3b8" };

        const token = document.createElement("span");
        token.className = "token";
        token.style.setProperty("--tag-color", meta.color);
        token.title = `${tag} — ${meta.label}`;

        const wordSpan = document.createElement("span");
        wordSpan.className = "token-word";
        wordSpan.textContent = word;

        const tagSpan = document.createElement("span");
        tagSpan.className = "token-tag";
        tagSpan.textContent = tag;

        token.append(wordSpan, tagSpan);
        tokens.append(token);
        tokenCount += 1;
      });

      const meta = document.createElement("p");
      meta.className = "result-meta";
      meta.textContent = `${result.tokens.length} tokens`;

      article.append(tokens, meta);
      fragment.append(article);
    }

    els.output.replaceChildren(fragment);
    els.timing.textContent = `${tokenCount} tokens · ${elapsedMs.toFixed(1)} ms`;
  }

  function tagInput() {
    if (!engine) return;

    const started = performance.now();
    const results = [];
    let tokenCount = 0;
    let truncated = false;

    for (const line of els.input.value.split(/\r?\n/)) {
      if (tokenCount >= MAX_TOKENS) break;

      const tokens = tokenize(line);
      if (tokens.length === 0) continue;

      if (tokenCount + tokens.length > MAX_TOKENS) {
        truncated = true;
        setStatus(
          `Long input truncated to the first ${MAX_TOKENS} tokens.`,
          false
        );
      }

      const limited = tokens.slice(0, MAX_TOKENS - tokenCount);
      results.push({ tokens: limited, tags: engine.tagTokens(limited) });
      tokenCount += limited.length;
    }

    render(results, performance.now() - started);

    if (results.length === 0) {
      els.timing.textContent = "";
    } else if (!truncated) {
      setStatus(readyStatus);
    }
  }

  function setReady(model) {
    const vocab = Number(model.meta?.vocab_size || 0);
    const alpha = Number(model.meta?.alpha || 0);

    setPill("ready", "Model ready");
    readyStatus = `Ready · ${vocab.toLocaleString("en-US")}-word vocabulary · ${model.tags.length} tags · α = ${alpha}`;
    setStatus(readyStatus);
    els.tag.disabled = false;
    els.input.disabled = false;

    tagInput();
  }

  function setFailed(message) {
    setPill("error", "Model unavailable");
    setStatus(message, true);
    clearOutput("The model could not be loaded.");
  }

  async function init() {
    renderLegend();
    renderExamples();
    clearOutput("Loading the model …");
    els.input.disabled = true;

    if (window.location.protocol === "file:") {
      setFailed(
        "Browsers block file:// requests. Serve this folder over HTTP (python -m http.server) or open the GitHub Pages deployment."
      );
      return;
    }

    try {
      const response = await fetch(MODEL_URL, { cache: "force-cache" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const model = await response.json();
      engine = createEngine(model);
      setReady(model);
    } catch (error) {
      console.error(error);
      setFailed(
        `Could not load ${MODEL_URL}. Check that docs/model/hmm_pos_model.json exists and that the site is served over HTTP.`
      );
    }
  }

  els.tag.addEventListener("click", tagInput);

  els.clear.addEventListener("click", () => {
    els.input.value = "";
    els.input.focus();
    clearOutput("Type a sentence to tag it.");
  });

  els.input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      tagInput();
    }
  });

  init();
})();
