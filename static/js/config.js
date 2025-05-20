let selectedModelData = null;
let modelConfig = {};

document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.getElementById('model_type');
  const topicInputGroup = document.getElementById('topicCountGroup');
 

  
  modelSelect.addEventListener('change', function () {
    const selected = this.value;
    topicInputGroup.style.display = selected === 'External' ? 'none' : 'block';
  });

  loadCorpusCheckboxTable();
  loadModelConfig();

  document.getElementById('modelSelectionForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const model = modelSelect.value;
    const topicNumber = document.getElementById("numTopics").value;
    

    renderParamsIntoModal(model);

    const modal = new bootstrap.Modal(document.getElementById('modelNameModal'));
    modal.show();

    document.getElementById('modelNameModal').addEventListener('shown.bs.modal', () => {
      document.getElementById('modelNameInput')?.focus();
    });
  });

  document.getElementById('confirmModelName').addEventListener('click', () => {
    const modelName = document.getElementById('modelNameInput').value.trim();
    const numTopics = document.getElementById('numTopics').value.trim();
    const fullscreenLoader = document.getElementById('fullscreenLoader');
    const corpus = document.getElementById("corpusName");
    

    if (!modelName) {
        alert("Please provide a model name.");
        return;
    }


    const selectedCorpora = Array.from(
      document.querySelectorAll('input[name="corpusName"]:checked')
    ).map(cb => cb.value);
    

    // Collect advanced settings
      const training_param = {};
      const inputs = document.querySelectorAll("#modelParamsArea input");
      inputs.forEach(input => {
          const key = input.id;
          const value = input.value.trim();
          training_param[key] = input.type === "number" ? Number(value) : value;
      });

      // Move num_topics into advanced settings
      training_param["num_topics"] = Number(document.getElementById("numTopics").value);

      const requestData = {
          model: modelSelect.value,               // Model selected
          save_name: modelName,                   // Model name
          training_params: training_param,    // ✅ Unified training params
          corpuses: selectedCorpora,
      };

    console.log(requestData)
    fullscreenLoader.style.display = 'flex';



    fetch('/train_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(async res => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'Model run failed.');
        bootstrap.Modal.getInstance(document.getElementById('modelNameModal')).hide();
        alert(data.message || 'Model run completed!');
        window.location.href = "/trained-models";

    })
    .catch(err => {
        alert("Error running model: " + err.message);
        console.error("Model run error:", err);
    })
    .finally(() => {
        fullscreenLoader.style.display = 'none';
    });
});


  const modalElement = document.getElementById('modelNameModal');
  modalElement.addEventListener('hidden.bs.modal', () => {
    if (modalElement.contains(document.activeElement)) {
      document.activeElement.blur();
    }
  });
});

async function loadCorpusCheckboxTable() {
  const tbody = document.getElementById("corpusCheckboxTable");
  tbody.innerHTML = `<tr><td colspan="3"><em>Loading corpora...</em></td></tr>`;

  try {
    const res = await fetch("/corpora");
    const corpora = await res.json();

    if (corpora.length > 0) {
      tbody.innerHTML = "";

      corpora.forEach((name, index) => {
        const row = document.createElement("tr");

        // Checkbox Cell
        const checkboxCell = document.createElement("td");
        checkboxCell.className = "align-middle";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "corpusName";
        checkbox.value = name;
        checkbox.className = "form-check-input";
        checkbox.id = `corpus-${index}`;
        checkboxCell.appendChild(checkbox);

        // Name Cell
        const nameCell = document.createElement("td");
        nameCell.className = "align-middle";
        const label = document.createElement("label");
        label.htmlFor = `corpus-${index}`;
        label.textContent = name;
        nameCell.appendChild(label);

        // Delete Button Cell
        const deleteCell = document.createElement("td");
        deleteCell.className = "text-end align-middle";
        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "btn btn-sm btn-outline-danger";
        deleteBtn.textContent = "Delete";
        deleteBtn.title = "Delete Corpus";
        deleteBtn.onclick = () => deleteCorpus(name);
        deleteCell.appendChild(deleteBtn);

        // Append all cells to row
        row.appendChild(checkboxCell);
        row.appendChild(nameCell);
        row.appendChild(deleteCell);
        tbody.appendChild(row);
      });

      // ✅ Enable Train button only if at least one checkbox is selected
      const trainBtn = document.getElementById("trainBtn");

      function updateTrainButtonState() {
        const anyChecked = document.querySelectorAll('input[name="corpusName"]:checked').length > 0;
        trainBtn.disabled = !anyChecked;
      }

      // Add listener to all checkboxes
      document.querySelectorAll('input[name="corpusName"]').forEach(cb => {
        cb.addEventListener("change", updateTrainButtonState);
      });

      // Initial check
      updateTrainButtonState();
    } else {
      tbody.innerHTML = `<tr><td colspan="3"><em>No corpora found</em></td></tr>`;
    }
  } catch (error) {
    console.error("Error loading corpora:", error);
    tbody.innerHTML = `<tr><td colspan="3"><em>Error loading corpora</em></td></tr>`;
  }
}




async function deleteCorpus(name) {
  if (!confirm(`Are you sure you want to delete "${name}"?`)) return;

  try {
    const res = await fetch(`/delete-corpus/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ corpus_name: name })
    });

    const data = await res.json();

    if (!res.ok) {
      console.error("Server responded with error:", data);
      alert(`Error: ${data.message}`);
      return;
    }

    alert(data.message);
    loadCorpusCheckboxTable(); // reload the table after deletion
  } catch (err) {
    console.error("Fetch failed:", err);
    alert("Failed to delete the corpus.");
  }
}





async function loadModelConfig() {
  if (Object.keys(modelConfig).length) return;

  try {
    const res = await fetch("/static/config/config.yaml");
    const yamlText = await res.text();           // Read as text
    modelConfig = jsyaml.load(yamlText);          // Parse YAML into JS object
    console.log("Loaded model config:", modelConfig);
  } catch (error) {
    console.error("Error loading model config:", error);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const toggleButton = document.getElementById("toggleAdvanced");
  const advancedSection = document.getElementById("advancedSettings");

  toggleButton.addEventListener("click", () => {
      const isVisible = advancedSection.style.display === "block";
      advancedSection.style.display = isVisible ? "none" : "block";
      toggleButton.textContent = isVisible ? "Show Advanced Settings" : "Hide Advanced Settings";
  });
});


function renderParamsIntoModal(model) {
  const container = document.getElementById('modelParamsArea');
  container.innerHTML = '';

  const params = modelConfig.topic_modeling[model];  // Adjust based on your structure
  if (!params) return;

  for (const key in params) {
      const value = params[key];

      // Each input will take half-width (col-6)
      const col = document.createElement('div');
      col.className = 'col-md-6'; // You can change to 'col-md-4' if you want 3 columns

      const label = document.createElement('label');
      label.textContent = key.replace(/_/g, " ");
      label.setAttribute('for', key);
      label.classList.add('form-label');

      const input = document.createElement('input');
      input.type = typeof value === 'number' ? 'number' : 'text';
      input.className = 'form-control';
      input.id = key;
      input.name = key;
      input.value = value ?? '';

      col.appendChild(label);
      col.appendChild(input);
      container.appendChild(col);
  }
}