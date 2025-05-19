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
  tbody.innerHTML = `<tr><td colspan="2"><em>Loading corpora...</em></td></tr>`;

  try {
    const res = await fetch("/corpora");
    const corpora = await res.json();

    if (corpora.length > 0) {
      tbody.innerHTML = "";

      corpora.forEach((name, index) => {
        const row = document.createElement("tr");

        const checkboxCell = document.createElement("td");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "corpusName";
        checkbox.value = name;
        checkbox.className = "form-check-input";
        checkbox.id = `corpus-${index}`;
        checkboxCell.appendChild(checkbox);

        const nameCell = document.createElement("td");
        const label = document.createElement("label");
        label.htmlFor = `corpus-${index}`;
        label.textContent = name;
        nameCell.appendChild(label);

        row.appendChild(checkboxCell);
        row.appendChild(nameCell);
        tbody.appendChild(row);
      });
    } else {
      tbody.innerHTML = `<tr><td colspan="2"><em>No corpora found</em></td></tr>`;
    }
  } catch (error) {
    console.error("Error loading corpora:", error);
    tbody.innerHTML = `<tr><td colspan="2"><em>Error loading corpora</em></td></tr>`;
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