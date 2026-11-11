/* UI wiring for the HMM POS Tagger demo. */
(function () {
  "use strict";

  const { createEngine, tokenize, TAG_META } = window.HMMTagger;
  const MODEL_URL = "model/hmm_pos_model.json";
  const MAX_TOKENS = 600;
  const TAG_ORDER = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", ".", "X"];

  const EXAMPLES = [
    { label: "Quick fox", text: "The quick brown fox jumps over the lazy dog ." },
    { label: "Ambiguity", text: "It was a very tempting offer ." },
    { label: "Unseen words", text: "The blockchain startup open-sourced its transformer pipeline in Phnom Penh ." },
    { label: "Long sentence", text: "According to state law a slave had to be at least thirty years old before he could be freed ." },
    { label: "Numbers", text: "With tips , the girls average between $150 and $200 a week ." },
  ];

  const els = {
    input: document.getElementById("inputText"),
    tag: document.getElementById("tagBtn"),
    clear: document.getElementById("clearBtn"),
    output: document.getElementById("output"),
    timing: document.getElementById("timing"),
    status: document.getElementById("status"),
    statusText: document.getElementById("statusText"),
    examples: document.getElementById("examples"),
    legend: document.getElementById("tagLegend"),
  };

  let engine = null;
  let readyText = "";

  function setStatus(state, text) {
    els.status.dataset.state = state;
    els.statusText.textContent = text;
  }

  function renderLegend() {
    const fragment = document.createDocumentFragment();

    for (const tag of TAG_ORDER) {
      const meta = TAG_META[tag];
      if (!meta) continue;

      const item = document.createElement("li");
      item.className = "tag-item";
      item.style.setProperty("--tag-color", meta.color);
      item.title = meta.description;

      const dot = document.createElement("span");
      dot.className = "tag-dot";
      dot.setAttribute("aria-hidden", "true");

      const code = document.createElement("strong");
      code.textContent = meta.short || tag;

      const name = document.createElement("span");
      name.className = "tag-name";
      name.textContent = meta.label;

      item.append(dot, code, name);
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

  function showPlaceholder(message) {
    els.output.replaceChildren();
    const placeholder = document.createElement("p");
    placeholder.className = "placeholder";
    placeholder.textContent = message;
    els.output.append(placeholder);
    els.timing.textContent = "";
  }

  function render(results, elapsedMs) {
    if (results.length === 0) {
      showPlaceholder("Type a sentence to tag it.");
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
        const meta = TAG_META[tag] || { label: tag, color: "#63686f" };

        const token = document.createElement("span");
        token.className = "token";
        token.style.setProperty("--tag-color", meta.color);
        token.title = `${tag} — ${meta.label}`;

        const wordSpan = document.createElement("span");
        wordSpan.className = "token-word";
        wordSpan.textContent = word;

        const tagSpan = document.createElement("span");
        tagSpan.className = "token-tag";
        tagSpan.textContent = meta.short || tag;

        token.append(wordSpan, tagSpan);
        tokens.append(token);
        tokenCount += 1;
      });

      article.append(tokens);
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
      if (tokenCount >= MAX_TOKENS) {
        truncated = true;
        break;
      }

      const tokens = tokenize(line);
      if (tokens.length === 0) continue;

      if (tokenCount + tokens.length > MAX_TOKENS) {
        truncated = true;
      }

      const limited = tokens.slice(0, MAX_TOKENS - tokenCount);
      results.push({ tokens: limited, tags: engine.tagTokens(limited) });
      tokenCount += limited.length;
    }

    render(results, performance.now() - started);

    if (results.length === 0) {
      setStatus("ready", readyText);
      els.timing.textContent = "";
    } else if (truncated) {
      setStatus("notice", `Showing the first ${MAX_TOKENS} tokens.`);
    } else {
      setStatus("ready", readyText);
    }
  }

  function setReady(model) {
    const vocab = Number(model.meta?.vocab_size || 0);
    const alpha = Number(model.meta?.alpha || 0);

    readyText = `Model ready · ${vocab.toLocaleString("en-US")} words · α = ${alpha}`;
    setStatus("ready", readyText);
    els.tag.disabled = false;

    tagInput();
  }

  function setFailed(message) {
    setStatus("error", message);
    showPlaceholder("The model could not be loaded.");
    els.tag.disabled = true;
  }

  async function init() {
    renderLegend();
    renderExamples();
    showPlaceholder("Loading the model …");

    if (window.location.protocol === "file:") {
      setFailed("Browsers block file:// requests — serve this folder over HTTP.");
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
      setFailed(`Could not load ${MODEL_URL}. Check that the file exists and is served over HTTP.`);
    }
  }

  els.tag.addEventListener("click", tagInput);

  els.clear.addEventListener("click", () => {
    els.input.value = "";
    els.input.focus();
    showPlaceholder("Type a sentence to tag it.");
    if (engine) setStatus("ready", readyText);
  });

  els.input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      tagInput();
    }
  });

  init();
})();
