// file-validation.js

let files = null;
let previewFiles = [];
let previewIndex = 0;
const allowedExtensions = ['csv', 'xls', 'xlsx', 'json', 'jsonl', 'txt'];
const tooltip = document.getElementById('hoverTooltip');

// Preview first 6 files

document.getElementById('files').addEventListener('change', async function (e) {
    previewFiles = Array.from(e.target.files).slice(0, 6);
    previewIndex = 0;

    if (previewFiles.length === 0) {
        alert("No files selected.");
        return;
    }

    await showPreview(previewIndex);
    const modal = new bootstrap.Modal(document.getElementById('filePreviewModal'));
    modal.show();
});

// Display logic based on extension
async function showPreview(index) {
    const file = previewFiles[index];
    const extension = file.name.split('.').pop().toLowerCase();
    const previewContainer = document.getElementById('preview-content');
    previewContainer.innerHTML = '';

    document.getElementById('filePreviewModalLabel').textContent = `Preview - ${file.name}`;
    document.getElementById('preview-counter').textContent = `File ${index + 1} of ${previewFiles.length}`;

    try {
        if (['csv'].includes(extension)) {
            const content = await file.text();
            const parsed = Papa.parse(content, { header: true });
            const headers = parsed.meta.fields || [];
            const rows = parsed.data.slice(0, 5);

            if (headers.length) {
                const table = buildTable(headers, rows);
                previewContainer.appendChild(table);
            } else {
                previewContainer.innerHTML = '<p class="text-muted">No valid CSV data found.</p>';
            }
        } else if (extension === 'json') {
            const content = await file.text();
            const jsonData = JSON.parse(content);
            previewContainer.innerHTML = `<pre>${JSON.stringify(Array.isArray(jsonData) ? jsonData.slice(0, 5) : jsonData, null, 2)}</pre>`;
        } else if (extension === 'jsonl') {
            const lines = (await file.text()).trim().split('\n').slice(0, 5);
            const parsed = lines.map(line => JSON.parse(line));
            previewContainer.innerHTML = `<pre>${JSON.stringify(parsed, null, 2)}</pre>`;
        } else if (extension === 'txt') {
            const content = await file.text();
            const lines = content.split('\n').slice(0, 20);
            previewContainer.innerHTML = `<pre>${lines.join('\n')}</pre>`;
        } else if (['xls', 'xlsx'].includes(extension)) {
            const arrayBuffer = await file.arrayBuffer();
            const workbook = XLSX.read(arrayBuffer, { type: "array" });
            const firstSheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[firstSheetName];
            const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

            if (jsonData.length) {
                const headers = jsonData[0];
                const rows = jsonData.slice(1, 6);
                const table = buildTable(headers, rows);
                previewContainer.appendChild(table);
            } else {
                previewContainer.innerHTML = '<p class="text-muted">No valid Excel data found.</p>';
            }
        } else {
            previewContainer.innerHTML = '<p class="text-muted">Unsupported format.</p>';
        }
    } catch (err) {
        previewContainer.innerHTML = `<pre class='text-danger'>Preview failed: ${err.message}</pre>`;
    }

    document.getElementById('prevFile').style.display = index === 0 ? 'none' : 'inline-block';
    document.getElementById('nextFile').style.display = index === previewFiles.length - 1 ? 'none' : 'inline-block';
}

function buildTable(headers, rows) {
    const table = document.createElement('table');
    table.className = 'table table-bordered table-striped table-sm';
    table.innerHTML = `
        <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
        <tbody>
            ${rows.map(row => `<tr>${headers.map((_, i) => `<td>${row[i] || ''}</td>`).join('')}</tr>`).join('')}
        </tbody>
    `;
    return table;
}

document.getElementById('prevFile').addEventListener('click', () => {
    if (previewIndex > 0) showPreview(--previewIndex);
});

document.getElementById('nextFile').addEventListener('click', () => {
    if (previewIndex < previewFiles.length - 1) showPreview(++previewIndex);
});
