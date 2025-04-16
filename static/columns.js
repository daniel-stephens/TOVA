document.getElementById("files").addEventListener("change", getColumns);

async function getColumns(event) {
  const files = event.target.files;
  const columnsSet = new Set();
  let isAllTxt = true;

  for (const file of files) {
    const fileExt = file.name.split('.').pop().toLowerCase();
    const text = await file.text();

    if (fileExt === "csv") {
      isAllTxt = false;
      const firstLine = text.split('\n')[0];
      const headers = firstLine.split(',').map(h => h.trim().replace(/^"|"$/g, ''));
      headers.forEach(col => columnsSet.add(col));
    } else if (["json", "jsonl"].includes(fileExt)) {
      isAllTxt = false;
      try {
        const lines = fileExt === "jsonl" ? text.split('\n') : [text];
        const firstObj = JSON.parse(lines.find(line => line.trim()));
        Object.keys(firstObj).forEach(col => columnsSet.add(col));
      } catch (err) {
        console.error("JSON parse error:", err);
      }
    } else if (["xls", "xlsx"].includes(fileExt)) {
      isAllTxt = false;
      console.log("Excel file detected — requires backend parsing or SheetJS for client-side parsing.");
      continue;
    } else if (fileExt === "txt") {
      continue; // still assume txt means "content"
    } else {
      isAllTxt = false;
    }
  }

  const textSelect = document.getElementById("textColumn");
  const labelSelect = document.getElementById("labelColumn");

  // Reset options
  textSelect.innerHTML = '<option value="" disabled selected>Select text column</option>';
  labelSelect.innerHTML = '<option value="" disabled selected>Select label column (optional)</option>';

  if (isAllTxt) {
    textSelect.innerHTML += `<option value="content" selected>content</option>`;
    labelSelect.innerHTML += `<option value="">(none)</option>`;
  } else {
    columnsSet.forEach(col => {
      textSelect.innerHTML += `<option value="${col}">${col}</option>`;
      labelSelect.innerHTML += `<option value="${col}">${col}</option>`;
    });
  }
}
