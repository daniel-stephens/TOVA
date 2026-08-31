/* Active learning, review-first layout (prototype).
 *
 * Extracted from the template so it can be cached, linted and eventually
 * tested -- active_learning_two.html carries ~1,716 lines of JS inline across
 * 127 functions, which is the reason that page is hard to change.
 *
 * Talks to exactly the same endpoints as active_learning_two; no backend
 * change. Endpoints and the model id arrive via data- attributes rather than
 * template interpolation, which is what lets this be a static file.
 *
 * Deliberately partial: bulk labelling, the "Similar" view and the preview
 * modal are not implemented. This exists to make the layout and the
 * one-keystroke loop concrete enough to judge.
 */
(function () {
  "use strict";

  var root = document.getElementById("al3");
  if (!root) { return; }

  var MODEL_ID = root.dataset.modelId || "";
  var URLS = {
    recommend: root.dataset.urlRecommend,
    label: root.dataset.urlLabel,
    labels: root.dataset.urlLabels,
  };

  var docs = [];         // [{id, text, suggestions:[{label,score}], label}]
  var index = 0;
  var humanLabels = {};  // doc id -> label the participant applied
  var selected = "";
  var filter = "";

  var el = {
    body: document.getElementById("al3Body"),
    docId: document.getElementById("al3DocId"),
    current: document.getElementById("al3Current"),
    chips: document.getElementById("al3Chips"),
    input: document.getElementById("al3Input"),
    save: document.getElementById("al3Save"),
    skip: document.getElementById("al3Skip"),
    back: document.getElementById("al3Back"),
    feedback: document.getElementById("al3Feedback"),
    list: document.getElementById("al3List"),
    search: document.getElementById("al3Search"),
    count: document.getElementById("al3Count"),
    total: document.getElementById("al3Total"),
    fill: document.getElementById("al3Fill"),
    banner: document.getElementById("al3Banner"),
  };

  function csrf() {
    var m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    return m ? decodeURIComponent(m[1]) : "";
  }

  function getJSON(url) {
    return fetch(url, { headers: { "X-Requested-With": "XMLHttpRequest" } })
      .then(function (r) {
        if (!r.ok) { throw new Error("Request failed (" + r.status + ")"); }
        return r.json();
      });
  }

  function postJSON(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrf(),
        "X-Requested-With": "XMLHttpRequest",
      },
      body: JSON.stringify(payload),
    }).then(function (r) {
      if (!r.ok) { throw new Error("Save failed (" + r.status + ")"); }
      return r.json().catch(function () { return {}; });
    });
  }

  /* The recommend endpoint is proxied straight from the backend, so be liberal
     about the shape it comes back in rather than assuming one. */
  function normalise(payload) {
    var raw = payload;
    if (!Array.isArray(raw)) {
      raw = payload.documents || payload.docs || payload.recommendations || [];
    }
    return raw.map(function (d, i) {
      var suggestions = d.suggestions || d.labels || d.candidates || [];
      return {
        id: String(d.id != null ? d.id : (d.doc_id != null ? d.doc_id : i)),
        displayId: String(d.original_id || d.display_id || d.id || i),
        text: d.raw_text || d.text || d.body || "",
        label: d.label || d.model_label || "",
        suggestions: suggestions.map(function (s) {
          return typeof s === "string"
            ? { label: s, score: null }
            : { label: s.label || s.name || "", score: s.score != null ? s.score : s.confidence };
        }).filter(function (s) { return s.label; }),
      };
    });
  }

  function current() { return docs[index] || null; }

  function labelFor(doc) {
    return humanLabels[doc.id] || doc.label || "";
  }

  function setFeedback(msg, isError) {
    el.feedback.textContent = msg || "";
    el.feedback.classList.toggle("is-error", !!isError);
  }

  function renderProgress() {
    var done = docs.filter(function (d) { return humanLabels[d.id]; }).length;
    el.count.textContent = String(done);
    el.total.textContent = String(docs.length);
    el.fill.style.width = docs.length ? (100 * done / docs.length) + "%" : "0%";
    el.banner.classList.toggle("is-hidden", !(docs.length && done === docs.length));
  }

  function renderChips() {
    var doc = current();
    el.chips.innerHTML = "";
    if (!doc) { return; }

    var suggestions = doc.suggestions.slice(0, 9);
    if (!suggestions.length) {
      var none = document.createElement("span");
      none.className = "al3-chip__conf";
      none.textContent = "No suggestions for this document — type a label below.";
      el.chips.appendChild(none);
      return;
    }

    suggestions.forEach(function (s, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "al3-chip" + (selected === s.label ? " is-selected" : "");
      btn.dataset.label = s.label;
      // Announce the real total, not a hardcoded one.
      btn.setAttribute(
        "aria-label",
        "Suggestion " + (i + 1) + " of " + suggestions.length +
        " (press " + (i + 1) + "): " + s.label
      );

      var kbd = document.createElement("kbd");
      kbd.textContent = String(i + 1);
      btn.appendChild(kbd);

      var name = document.createElement("span");
      name.textContent = s.label;
      btn.appendChild(name);

      if (s.score != null) {
        var conf = document.createElement("span");
        conf.className = "al3-chip__conf";
        conf.textContent = Math.round(s.score * 100) + "%";
        btn.appendChild(conf);
      }

      btn.addEventListener("click", function () { pick(s.label); });
      el.chips.appendChild(btn);
    });
  }

  function renderDoc() {
    var doc = current();
    if (!doc) {
      el.docId.textContent = "—";
      el.body.textContent = "Nothing left to label.";
      el.chips.innerHTML = "";
      el.current.textContent = "";
      el.save.disabled = true;
      return;
    }
    el.docId.textContent = doc.displayId;
    el.body.textContent = doc.text || "(no text available)";
    el.body.scrollTop = 0;

    var applied = labelFor(doc);
    el.current.innerHTML = applied
      ? "Current label: <b></b>"
      : "<span>Unlabeled</span>";
    if (applied) { el.current.querySelector("b").textContent = applied; }

    selected = humanLabels[doc.id] || "";
    el.input.value = selected;
    renderChips();
    updateSave();
  }

  function renderList() {
    el.list.innerHTML = "";
    var matches = docs.filter(function (d) {
      if (!filter) { return true; }
      var f = filter.toLowerCase();
      return d.displayId.toLowerCase().indexOf(f) !== -1
        || (labelFor(d) || "").toLowerCase().indexOf(f) !== -1;
    });

    if (!matches.length) {
      var empty = document.createElement("p");
      empty.className = "al3-empty";
      empty.textContent = "No matching documents";
      el.list.appendChild(empty);
      return;
    }

    matches.forEach(function (d) {
      var i = docs.indexOf(d);
      var row = document.createElement("button");
      row.type = "button";
      row.className = "al3-row" + (i === index ? " is-current" : "");

      var id = document.createElement("span");
      id.className = "al3-row__id";
      id.textContent = d.displayId;
      row.appendChild(id);

      var lbl = document.createElement("span");
      var applied = humanLabels[d.id];
      lbl.className = "al3-row__label"
        + (applied ? " is-human" : (d.label ? "" : " is-unlabeled"));
      lbl.textContent = applied || d.label || "Unlabeled";
      row.appendChild(lbl);

      row.addEventListener("click", function () { go(i); });
      el.list.appendChild(row);
    });
  }

  function render() {
    renderDoc();
    renderList();
    renderProgress();
  }

  function updateSave() {
    el.save.disabled = !(el.input.value || "").trim();
  }

  function pick(label) {
    selected = label;
    el.input.value = label;
    renderChips();
    updateSave();
  }

  function go(i) {
    if (i < 0 || i >= docs.length) { return; }
    index = i;
    setFeedback("");
    render();
  }

  /* Advance to the next document that has no human label yet, so the queue
     drains rather than looping over work already done. */
  function nextUnlabeled() {
    for (var step = 1; step <= docs.length; step++) {
      var i = (index + step) % docs.length;
      if (!humanLabels[docs[i].id]) { return i; }
    }
    return index;
  }

  /* The whole loop in one keystroke: save this label and move on. */
  function saveAndAdvance() {
    var doc = current();
    var label = (el.input.value || "").trim();
    if (!doc || !label) { return; }

    el.save.disabled = true;
    setFeedback("Saving…");
    postJSON(URLS.label, { model_id: MODEL_ID, doc_id: doc.id, label: label })
      .then(function () {
        humanLabels[doc.id] = label;
        setFeedback("Saved “" + label + "”");
        go(nextUnlabeled());
      })
      .catch(function (err) {
        setFeedback(err.message || "Could not save that label.", true);
        el.save.disabled = false;
      });
  }

  el.save.addEventListener("click", saveAndAdvance);
  el.skip.addEventListener("click", function () { go((index + 1) % docs.length); });
  el.back.addEventListener("click", function () { go(index > 0 ? index - 1 : docs.length - 1); });
  el.input.addEventListener("input", updateSave);
  el.search.addEventListener("input", function () { filter = el.search.value; renderList(); });

  el.input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); saveAndAdvance(); }
  });

  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    var t = e.target;
    var typing = t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable;

    if (e.key === "/" && !typing) { e.preventDefault(); el.search.focus(); return; }
    if (typing) { return; }

    if (e.key === "Enter") { e.preventDefault(); saveAndAdvance(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go((index + 1) % docs.length); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(index > 0 ? index - 1 : docs.length - 1); }
    else if (e.key >= "1" && e.key <= "9") {
      var doc = current();
      if (!doc) { return; }
      var s = doc.suggestions[Number(e.key) - 1];
      if (s) { e.preventDefault(); pick(s.label); }
    }
  });

  if (!MODEL_ID) {
    el.body.textContent = "No model selected.";
    return;
  }

  Promise.all([
    getJSON(URLS.recommend + "?model_id=" + encodeURIComponent(MODEL_ID)),
    getJSON(URLS.labels + "?model_id=" + encodeURIComponent(MODEL_ID))
      .catch(function () { return {}; }),
  ])
    .then(function (results) {
      docs = normalise(results[0] || {});
      humanLabels = (results[1] && results[1].labels) || {};
      if (!docs.length) {
        el.body.textContent = "No documents came back for this model.";
      }
      index = docs.length ? nextUnlabeled() : 0;
      render();
    })
    .catch(function (err) {
      el.body.textContent = err.message || "Could not load documents.";
      setFeedback("", false);
    });
})();
