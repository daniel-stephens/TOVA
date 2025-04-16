document.addEventListener("DOMContentLoaded", function () {
    const uploadButton = document.getElementById("upload");
    const fileInput = document.getElementById("file-upload");
    const fileList = document.getElementById("file-list");

    // Ensure pdf.js library initialization
    if (!window.pdfjsLib) {
        console.error("PDF.js library not loaded. Ensure it's correctly imported.");
        return;
    }

    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.12.313/pdf.worker.min.js';

    uploadButton.addEventListener("click", async function (event) {
        event.preventDefault();

        if (fileInput.files.length === 0) {
            alert("❌ No files selected!");
            return;
        }

        let selectedFileType = document.querySelector('input[name="fileType"]:checked').value;
        let files = Array.from(fileInput.files);

        let invalidFiles = files.filter(file => !validateFile(file, selectedFileType));

        if (invalidFiles.length > 0) {
            alert(`❌ Invalid files for selected type (${selectedFileType}):\n${invalidFiles.map(f => f.name).join(", ")}`);
            return;
        }

        if (selectedFileType === "pdf") {
            try {
                let extractedJSON = await processPDFs(files);
                console.log("Extracted PDF JSON:", extractedJSON);
                uploadExtractedPDFJSON(extractedJSON);
            } catch (err) {
                alert(`❌ PDF parsing failed: ${err}`);
            }
        } else {
            uploadFiles(files);
        }
    });

    function validateFile(file, selectedFileType) {
        const ext = file.name.split('.').pop().toLowerCase();
        const allowedTypes = {
            csv: ['csv'],
            json: ['json'],
            jsonl: ['jsonl'],
            excel: ['xls', 'xlsx'],
            pdf: ['pdf']
        };
        return allowedTypes[selectedFileType].includes(ext);
    }

    async function processPDFs(files) {
        let pdfJSON = [];
        for (let i = 0; i < files.length; i++) {
            let text = await extractTextFromPDF(files[i]);
            pdfJSON.push({
                document_number: i + 1,
                text: text,
            });
        }
        return pdfJSON;
    }

    async function extractTextFromPDF(file) {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        let textContent = "";

        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
            const page = await pdf.getPage(pageNum);
            const content = await page.getTextContent();
            const strings = content.items.map(item => item.str).join(" ");
            textContent += strings + "\n";
        }

        return textContent.trim();
    }

    function uploadFiles(files) {
        let formData = new FormData();
        files.forEach((file, index) => {
            formData.append(`file${index}`, file);
        });

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`❌ ${data.error}`);
            } else {
                alert(`✅ ${data.message}`);
                updateFileList(files);
                fileInput.value = "";
            }
        })
        .catch(() => {
            alert("❌ Error uploading files. Please try again.");
        });
    }

    function uploadExtractedPDFJSON(pdfData) {
        fetch("/upload", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pdf_data: pdfData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`❌ ${data.error}`);
            } else {
                alert(`✅ PDF text successfully uploaded and processed.`);
            }
        })
        .catch(() => {
            alert("❌ Error uploading extracted PDF text.");
        });
    }

    function updateFileList(files) {
        fileList.innerHTML = "";
        files.forEach(file => {
            let li = document.createElement("li");
            li.className = "list-group-item d-flex align-items-center";
            li.innerHTML = `<i class="bi bi-file-earmark-text"></i> <span>${file.name}</span>`;
            fileList.appendChild(li);
        });
    }
});
