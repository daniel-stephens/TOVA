// infer_page_logic.js

async function loadModelOptions() {
  const select = document.getElementById("modelDropdown");
  select.innerHTML = "<option selected disabled>Loading models...</option>";

  try {
    const res = await fetch("/get_models");
    const models = await res.json();

    select.innerHTML = "<option selected disabled>Select model</option>";
    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      select.appendChild(option);
    });
  } catch (err) {
    console.error("Failed to load models:", err);
    select.innerHTML = "<option disabled>Error loading models</option>";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadModelOptions();

  const inputTypeRadios = document.querySelectorAll('input[name="inputType"]');
  const textGroup = document.getElementById("textInputGroup");
  const fileGroup = document.getElementById("fileInputGroup");

  inputTypeRadios.forEach(r => {
    r.addEventListener("change", () => {
      const isText = r.value === "text";
      textGroup.style.display = isText ? "block" : "none";
      fileGroup.style.display = isText ? "none" : "block";
    });
  });

  document.getElementById("files").addEventListener("change", getColumns);

  document.getElementById("inferBtn").addEventListener("click", async () => {
    const model = document.getElementById("modelDropdown").value;
    const inputMethod = document.querySelector("input[name='inputType']:checked").value;
    const topicBox = document.getElementById("topicResults");

    topicBox.innerHTML = "<div class='text-muted text-center'>Loading...</div>";

    if (inputMethod === "text") {
      const text = document.getElementById("inputText").value.trim();
      if (!text || !model) {
        alert("Please enter text and select a model.");
        return;
      }
      const res = await fetch("/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model })
      });
      const result = await res.json();
      renderTopics(result.topics);

    } else {
      const file = document.getElementById("files").files[0];
      const text_col = document.getElementById("textColumn").value;
      const id_col = document.getElementById("idColumn").value;

      if (!file || !text_col || !id_col || !model) {
        alert("Please upload a file and select all columns.");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);
      formData.append("model", model);
      formData.append("text_col", text_col);
      formData.append("id_col", id_col);

      const res = await fetch("/infer-file", {
        method: "POST",
        body: formData
      });
      const result = await res.json();
      renderTopics(result.topics);
    }
  });
});

let fullTopicHTML = "";  // Global variable

function renderTopics(topics) {
const topicBox = document.getElementById("topicResults");
topicBox.innerHTML = "";

const isSingle = topics.length > 0 && !topics[0].doc_id;
let html = "";

if (isSingle) {
  const textId = "singleTextToggle";
  const inputText = topics[0].text || "";

  html = `
    <div class="mb-3 border rounded p-3 shadow-sm topic-item" data-text="${inputText.toLowerCase()}">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h6 class="fw-bold mb-0">Inferred Topics</h6>
        <button class="btn btn-sm btn-outline-info" onclick="toggleText('${textId}')">View Text</button>
      </div>
      <p id="${textId}" class="text-muted small" style="display: none;">${inputText}</p>

      ${topics.map((t) => `
        <div class="mb-2">
          <div class="d-flex justify-content-between align-items-center">
            <span>${t.label}</span>
            <span class="badge bg-primary">${(t.score * 100).toFixed(1)}%</span>
          </div>
          <div class="progress my-1" style="height: 6px;">
            <div class="progress-bar bg-primary" role="progressbar" style="width: ${t.score * 100}%"></div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
} else {
  html = topics.map((docResult, index) => {
    const textId = `textToggle${index}`;
    const text = docResult.text || "";
    return `
      <div class="mb-4 border rounded p-3 shadow-sm topic-item" data-text="${text.toLowerCase()}">
        <div class="d-flex justify-content-between align-items-center">
          <h6 class="fw-bold mb-0">Document ID: ${docResult.doc_id}</h6>
          <button class="btn btn-sm btn-outline-info" type="button" onclick="toggleText('${textId}')">
            View Text
          </button>
        </div>
        <p id="${textId}" class="text-muted small mt-2" style="display: none;">${text}</p>
        ${docResult.top_topics.map((t) => `
          <div class="mb-2">
            <div class="d-flex justify-content-between align-items-center">
              <span>${t.label}</span>
              <span class="badge bg-success">${(t.score * 100).toFixed(1)}%</span>
            </div>
            <div class="progress my-1" style="height: 6px;">
              <div class="progress-bar bg-success" role="progressbar" style="width: ${t.score * 100}%"></div>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }).join('');
}

topicBox.innerHTML = html;
fullTopicHTML = html;  // Save full content for filtering
}



// Utility to toggle text visibility
function toggleText(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = el.style.display === "none" ? "block" : "none";
}


async function getColumns(event) {
  const files = event.target.files;
  const columnsSet = new Set();
  let isAllTxt = true;

  for (const file of files) {
    if (file.name.startsWith("~$")) continue;
    const ext = file.name.split(".").pop().toLowerCase();
    const text = await file.text();

    if (ext === "csv") {
      isAllTxt = false;
      const firstLine = text.split('\n')[0];
      const headers = firstLine.split(',').map(h => h.trim().replace(/^"|"$/g, ''));
      headers.forEach(h => columnsSet.add(h));
    } else if (["json", "jsonl"].includes(ext)) {
      isAllTxt = false;
      try {
        const lines = ext === "jsonl" ? text.split('\n') : [text];
        const firstObj = JSON.parse(lines.find(l => l.trim()));
        Object.keys(firstObj).forEach(k => columnsSet.add(k));
      } catch (e) {
        console.error("JSON parse error:", e);
      }
    } else {
      continue;
    }
  }
  
    

  const textSelect = document.getElementById("textColumn");
  const idSelect = document.getElementById("idColumn");
  textSelect.innerHTML = '<option disabled selected>Select text column</option>';
  idSelect.innerHTML = '<option disabled selected>Select id column</option>';

  columnsSet.forEach(col => {
    textSelect.innerHTML += `<option value="${col}">${col}</option>`;
    idSelect.innerHTML += `<option value="${col}">${col}</option>`;
  });

  console.log("Detected columns:", [...columnsSet]);
}
