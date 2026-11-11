/*
 * tagger.js — dependency-free log-space Viterbi decoder.
 *
 * The exported model stores raw emission counts plus α and the vocabulary size,
 * so the browser can rebuild exactly the same smoothed B table as the notebook:
 *
 *   B(tag, w)   = (count(tag, w) + α) / (count(tag) + α(|V| + 1))
 *   B(tag, OOV) =  α                / (count(tag) + α(|V| + 1))
 *
 * Missing (word, tag) pairs therefore receive the OOV probability — identical
 * to the full smoothed table — which keeps the JSON artifact under 1 MB.
 */
(function (global) {
  "use strict";

  const DEFAULT_EPSILON = 1e-12;
  const TOKEN_RE = /[\p{L}\p{N}]+(?:['\u2019-][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu;

  const TAG_META = {
    NOUN: { label: "Noun", description: "Common and proper nouns", color: "#2f5d8a" },
    VERB: { label: "Verb", description: "All verb forms", color: "#a8452f" },
    ADJ: { label: "Adjective", description: "Modifiers of nouns", color: "#8a6510" },
    ADV: { label: "Adverb", description: "Modifiers of verbs and adjectives", color: "#3f6b35" },
    PRON: { label: "Pronoun", description: "Personal, relative, indefinite", color: "#67508f" },
    DET: { label: "Determiner", description: "Articles and quantifiers", color: "#9c3f60" },
    ADP: { label: "Adposition", description: "Prepositions and postpositions", color: "#226b66" },
    NUM: { label: "Numeral", description: "Numbers and numeric words", color: "#82592a" },
    CONJ: { label: "Conjunction", description: "Coordinating and subordinating", color: "#7d6a1f" },
    PRT: { label: "Particle", description: "Infinitival 'to', verb particles", color: "#4f5f8a" },
    ".": { label: "Punctuation", short: "PUNCT", description: "Sentence and clause boundaries", color: "#63686f" },
    X: { label: "Other", description: "Foreign words and abbreviations", color: "#8a5387" },
  };

  function tokenize(text) {
    return text.match(TOKEN_RE) || [];
  }

  function createEngine(model) {
    const tags = model.tags.slice();
    const stateCount = tags.length;
    const meta = model.meta || {};
    const alpha = Number(meta.alpha ?? 1e-9);
    const vocabSize = Number(meta.vocab_size ?? 0);
    const epsilon = Number(meta.epsilon ?? DEFAULT_EPSILON);

    const logStart = new Float64Array(stateCount);
    const logOov = new Float64Array(stateCount);
    const emissions = new Array(stateCount);
    const transitions = Array.from({ length: stateCount }, () => new Float64Array(stateCount));

    tags.forEach((tag, index) => {
      logStart[index] = Math.log(Number(model.pi[tag] ?? 0) + epsilon);
    });

    for (let i = 0; i < stateCount; i += 1) {
      const row = model.A[tags[i]] || {};
      for (let j = 0; j < stateCount; j += 1) {
        transitions[i][j] = Math.log(Number(row[tags[j]] ?? 0) + epsilon);
      }
    }

    for (let i = 0; i < stateCount; i += 1) {
      const tag = tags[i];
      const total = Number(model.tagTotals[tag] ?? 0);
      const denominator = total + alpha * (vocabSize + 1);
      const oovProbability = denominator > 0 ? alpha / denominator : 0;
      logOov[i] = Math.log(oovProbability + epsilon);

      const table = new Map();
      const counts = model.emissions[tag] || {};
      for (const word of Object.keys(counts)) {
        table.set(word, Math.log((Number(counts[word]) + alpha) / denominator + epsilon));
      }
      emissions[i] = table;
    }

    /**
     * Decode an array of tokens into an array of universal POS tags.
     * Mirrors `log_viterbi` in Mode.ipynb, including first-maximum tie-breaking.
     */
    function tagTokens(tokens) {
      const length = tokens.length;
      if (length === 0) return [];

      let previous = new Float64Array(stateCount);
      const firstWord = tokens[0].toLowerCase();

      for (let i = 0; i < stateCount; i += 1) {
        const emission = emissions[i].get(firstWord);
        previous[i] = logStart[i] + (emission === undefined ? logOov[i] : emission);
      }

      const backpointers = new Array(length);
      backpointers[0] = null;

      for (let t = 1; t < length; t += 1) {
        const word = tokens[t].toLowerCase();
        const emission = new Float64Array(stateCount);

        for (let j = 0; j < stateCount; j += 1) {
          const value = emissions[j].get(word);
          emission[j] = value === undefined ? logOov[j] : value;
        }

        const current = new Float64Array(stateCount);
        const back = new Uint8Array(stateCount);

        for (let j = 0; j < stateCount; j += 1) {
          let bestScore = -Infinity;
          let bestPrevious = 0;

          for (let i = 0; i < stateCount; i += 1) {
            const score = previous[i] + transitions[i][j];
            if (score > bestScore) {
              bestScore = score;
              bestPrevious = i;
            }
          }

          current[j] = bestScore + emission[j];
          back[j] = bestPrevious;
        }

        backpointers[t] = back;
        previous = current;
      }

      let best = 0;
      for (let j = 1; j < stateCount; j += 1) {
        if (previous[j] > previous[best]) best = j;
      }

      const path = new Array(length);
      path[length - 1] = best;
      for (let t = length - 1; t > 0; t -= 1) {
        best = backpointers[t][best];
        path[t - 1] = best;
      }

      return path.map((index) => tags[index]);
    }

    return { tags, meta, tagTokens, tokenize };
  }

  global.HMMTagger = { createEngine, tokenize, TAG_META };
})(typeof window !== "undefined" ? window : globalThis);
