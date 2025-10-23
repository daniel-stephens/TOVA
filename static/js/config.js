/* =========================
   Global state
========================= */
let selectedModelData = null;
let modelConfig = {};

/* =========================
   Helpers
========================= */
function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, m => (
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])
  ));
}

async function ensureModelConfigLoaded() {
  if (Object.keys(modelConfig || {}).length) return;
  await loadModelConfig();
}

/* =========================
   Boot
========================= */
document.addEventListener("DOMContentLoaded", () => {
  const modelSelect     = document.getElementById('model_type');
  const topicInputGroup = document.getElementById('topicCountGroup');
  const formEl          = document.getElementById('modelSelectionForm');
  const modalEl         = document.getElementById('modelNameModal');
  const confirmBtn      = document.getElementById('confirmModelName');
  const modal           = new bootstrap.Modal(modalEl);

  // Load dropdown + corpuses + config
  loadModelOptions();
  loadCorpusCheckboxTable();
  loadModelConfig();
  initModelNameGuard();     // optional; no-op if endpoint missing
  initAdvancedToggle();

  // Toggle topics input when model changes
  modelSelect.addEventListener('change', function () {
    topicInputGroup.style.display = this.value === 'External' ? 'none' : 'block';
  });
  // Set initial visibility
  topicInputGroup.style.display = modelSelect.value === 'External' ? 'none' : 'block';

  // Open modal on submit
  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Must have at least one corpus selected
    const checked = document.querySelectorAll('input[name="corpus"]:checked');
    if (checked.length === 0) {
      const trainBtn = document.getElementById("trainBtn");
      trainBtn?.classList.add("disabled");
      setTimeout(() => trainBtn?.classList.remove("disabled"), 400);
      return;
    }

    // Ensure config ready then render inputs
    await ensureModelConfigLoaded();
    const model = modelSelect.value;
    renderParamsIntoModal(model);

    modal.show();
  });

  // Focus model name once modal opens
  modalEl.addEventListener('shown.bs.modal', () => {
    document.getElementById('modelNameInput')?.focus();
  });

  // Confirm -> stash data + redirect
  confirmBtn.addEventListener('click', () => {
    const modelName = document.getElementById('modelNameInput').value.trim();
    const numTopics = document.getElementById('numTopics')?.value?.trim();

    if (!modelName) {
      alert("Please provide a model name.");
      return;
    }

    const selected = Array.from(document.querySelectorAll('input[name="corpus"]:checked')).map(cb => ({
      id: cb.dataset.id,
      name: cb.dataset.name,
      is_draft: cb.dataset.draft === '1'
    }));

    // Collect advanced params
    const training_param = {};
    document.querySelectorAll("#modelParamsArea input").forEach(input => {
      const key = input.id;
      const value = input.value;
      training_param[key] = input.type === "number" ? Number(value) : value;
    });

    // Append num_topics unless External hides it
    if (document.getElementById('topicCountGroup')?.style.display !== 'none') {
      training_param["num_topics"] = Number(numTopics || 10);
    }

    const requestData = {
      model: document.getElementById('model_type').value,
      save_name: modelName,
      training_params: training_param,
      corpuses: selected, // full objects (id, name, is_draft)
    };

    // Save for follow-up page
    sessionStorage.setItem("modelTrainingData", JSON.stringify(requestData));

    // Also pass a compact view via query params (names only)
    const query = new URLSearchParams({
      modelName: modelName,
      numTopics: training_param["num_topics"] ?? "",
      corpuses: selected.map(s => s.name).join(","),
    });
    window.location.href = `/training/?${query.toString()}`;
  });

  // Modal housekeeping
  modalEl.addEventListener('hidden.bs.modal', () => {
    if (modalEl.contains(document.activeElement)) {
      document.activeElement.blur();
    }
  });
});

/* =========================
   Model registry
========================= */
async function loadModelOptions() {
  const select = document.getElementById("model_type");
  try {
    const res = await fetch("/model-registry");
    if (!res.ok) throw new Error(`registry load failed: ${res.status}`);
    const models = await res.json();

    select.innerHTML = "";
    for (const [key] of Object.entries(models || {})) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = key;
      select.appendChild(option);
    }
    if (!select.options.length) {
      select.innerHTML = "<option disabled>No models available</option>";
    }
  } catch (err) {
    console.error("Error loading model options:", err);
    select.innerHTML = "<option disabled>Error loading models</option>";
  }
}

/* =========================
   Corpuses (drafts + saved)
========================= */
async function fetchAllCorpuses() {
  const res = await fetch("/getAllCorpora", { headers: { "Accept": "application/json" } });
  if (!res.ok) throw new Error(`Failed to load corpuses: ${res.status}`);
  const items = await res.json();
  // Expect [{id, name, is_draft, created_at}]
  return Array.isArray(items) ? items : [];
}

async function loadCorpusCheckboxTable() {
  const tbody = document.getElementById("corpusCheckboxTable");
  const trainBtn = document.getElementById("trainBtn");
  tbody.innerHTML = `<tr><td colspan="2"><em>Loading...</em></td></tr>`;

  try {
    const items = await fetchAllCorpuses();

    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="2"><em>No corpuses found.</em></td></tr>`;
      trainBtn.disabled = true;
      return;
    }

    const rows = items.map((it, idx) => {
      const safeName = escapeHtml(it.name || "");
      const badge = it.is_draft ? `<span class="badge rounded-pill text-bg-warning ms-2">Draft</span>` : "";
      const id = `corpus_${idx}`;
      return `
        <tr>
          <td class="text-center" style="width:80px;">
            <input
              type="checkbox"
              class="form-check-input"
              name="corpus"
              id="${id}"
              data-id="${escapeHtml(it.id || '')}"
              data-name="${safeName}"
              data-draft="${it.is_draft ? '1' : '0'}"
            >
          </td>
          <td>
            <label class="ms-1 mb-0" for="${id}">${safeName}${badge}</label>
          </td>
        </tr>`;
    });

    tbody.innerHTML = rows.join("");

    // Enable Train only if at least one selected
    const updateTrain = () => {
      const anyChecked = document.querySelectorAll('input[name="corpus"]:checked').length > 0;
      trainBtn.disabled = !anyChecked;
    };
    document.querySelectorAll('input[name="corpus"]').forEach(cb => {
      cb.addEventListener("change", updateTrain);
    });
    updateTrain();

  } catch (e) {
    console.error("Error loading corpuses:", e);
    tbody.innerHTML = `<tr><td colspan="2" class="text-danger"><em>Failed to load corpuses.</em></td></tr>`;
    trainBtn.disabled = true;
  }
}

/* =========================
   Delete corpus (optional)
   NOTE: drafts generally shouldn't be deleted here;
   this call expects a saved corpus name on your backend.
========================= */
async function deleteCorpus(name) {
  if (!confirm(`Are you sure you want to delete "${name}"?`)) return;

  try {
    const res = await fetch(`/delete-corpus/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corpus_name: name })
    });
    const data = await res.json();

    if (!res.ok) {
      console.error("Server responded with error:", data);
      alert(`Error: ${data.message || 'Delete failed'}`);
      return;
    }

    alert(data.message || "Deleted.");
    loadCorpusCheckboxTable();
  } catch (err) {
    console.error("Fetch failed:", err);
    alert("Failed to delete the corpus.");
  }
}

/* =========================
   Model config (YAML)
========================= */
async function loadModelConfig() {
  if (Object.keys(modelConfig).length) return;
  try {
    const res = await fetch("/static/config/config.yaml");
    const yamlText = await res.text();
    modelConfig = jsyaml.load(yamlText) || {};
    // console.log("Loaded model config:", modelConfig);
  } catch (error) {
    console.error("Error loading model config:", error);
  }
}

/* =========================
   Advanced toggle
========================= */
function initAdvancedToggle() {
  const toggleButton = document.getElementById("toggleAdvanced");
  const advancedSection = document.getElementById("advancedSettings");
  if (!toggleButton || !advancedSection) return;

  toggleButton.addEventListener("click", () => {
    const isVisible = advancedSection.style.display === "block";
    advancedSection.style.display = isVisible ? "none" : "block";
    toggleButton.textContent = isVisible ? "Show Advanced Settings" : "Hide Advanced Settings";
  });
}

/* =========================
   Duplicate model name guard (optional)
========================= */
async function initModelNameGuard() {
  const modelInput  = document.getElementById("modelNameInput");
  const warningText = document.getElementById("nameWarning");
  if (!modelInput || !warningText) return;

  let existingModelNames = [];
  try {
    const res = await fetch("/get_models");
    if (res.ok) existingModelNames = await res.json();
  } catch (err) {
    // Non-fatal
  }

  const confirmButton = document.getElementById("confirmModelName");
  const onChange = () => {
    const enteredName = modelInput.value.trim().toLowerCase();
    const dup = existingModelNames.some(n => String(n).toLowerCase() === enteredName);
    if (dup) {
      warningText.style.display = "block";
      modelInput.classList.add("is-invalid");
      if (confirmButton) confirmButton.disabled = true;
    } else {
      warningText.style.display = "none";
      modelInput.classList.remove("is-invalid");
      if (confirmButton) confirmButton.disabled = false;
    }
  };
  modelInput.addEventListener("input", onChange);
  onChange();
}

/* =========================
   Render params into modal
   Expects: modelConfig.topic_modeling[model]
========================= */
function renderParamsIntoModal(model) {
  const container = document.getElementById('modelParamsArea');
  container.innerHTML = '';

  const paramsRoot = (modelConfig && modelConfig.topic_modeling) || {};
  const params = paramsRoot[model];
  if (!params) return;

  for (const key of Object.keys(params)) {
    const value = params[key];

    const col = document.createElement('div');
    col.className = 'col-md-6';

    const label = document.createElement('label');
    label.textContent = key.replace(/_/g, " ");
    label.setAttribute('for', key);
    label.classList.add('form-label');

    const input = document.createElement('input');
    input.type = (typeof value === 'number') ? 'number' : 'text';
    input.className = 'form-control';
    input.id = key;
    input.name = key;
    input.value = value ?? '';

    col.appendChild(label);
    col.appendChild(input);
    container.appendChild(col);
  }
}
