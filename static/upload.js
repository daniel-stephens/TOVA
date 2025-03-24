document.addEventListener("DOMContentLoaded", function () {
    const uploadButton = document.querySelector(".btn-container button"); // Upload button
    const fileInput = document.getElementById("file-upload");
    const fileList = document.getElementById("file-list");

    uploadButton.addEventListener("click", async function (event) {
        event.preventDefault(); // Prevent default form submission

        if (fileInput.files.length === 0) {
            alert("❌ No files selected!");
            return;
        }

        let selectedFileType = document.querySelector('input[name="fileType"]:checked').value;
        let files = Array.from(fileInput.files);
        let invalidFiles = [];

        let validationPromises = files.map(file => validateFile(file, selectedFileType));

        Promise.all(validationPromises).then(async results => {
            invalidFiles = results.filter(result => !result.valid).map(result => result.file.name);

            if (invalidFiles.length > 0) {
                alert(`❌ The following files do not match the required structure:\n${invalidFiles.join(", ")}`);
            } else {
                if (selectedFileType === "pdf") {
                    let extractedJSON = await processPDFs(files);
                    console.log("Extracted PDF JSON:", extractedJSON);
                    alert("✅ PDF parsed successfully! Ready to upload.");
                    uploadExtractedPDFJSON(extractedJSON);
                } else {
                    uploadFiles(files);
                }
            }
        });
    });

    function validateFile(file, selectedFileType) {
        return new Promise(resolve => {
            let fileType = file.name.split('.').pop().toLowerCase();

            if (selectedFileType === "csv" && fileType !== "csv") {
                return resolve({ file, valid: false });
            }
            if (selectedFileType === "json" && fileType !== "json") {
                return resolve({ file, valid: false });
            }
            if (selectedFileType === "jsonl" && fileType !== "jsonl") {
                return resolve({ file, valid: false });
            }
            if (selectedFileType === "excel" && !["xls", "xlsx"].includes(fileType)) {
                return resolve({ file, valid: false });
            }
            if (selectedFileType === "pdf" && fileType !== "pdf") {
                return resolve({ file, valid: false });
            }

            resolve({ file, valid: true });
        });
    }

    async function processPDFs(files) {
        let pdfJSON = [];
        for (let i = 0; i < files.length; i++) {
            let file = files[i];
            let text = await extractTextFromPDF(file);
            pdfJSON.push({
                "document_number": i + 1,
                "text": text,
                "category": "PDF"
            });
        }
        return pdfJSON;
    }

    async function extractTextFromPDF(file) {
        return new Promise((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = async function () {
                const pdfData = new Uint8Array(reader.result);
                const pdf = await pdfjsLib.getDocument({ data: pdfData }).promise;
                let textContent = "";

                for (let i = 1; i <= pdf.numPages; i++) {
                    let page = await pdf.getPage(i);
                    let text = await page.getTextContent();
                    let pageText = text.items.map(item => item.str).join(" ");
                    textContent += `Page ${i}: ${pageText} \n`;
                }

                resolve(textContent.trim());
            };

            reader.onerror = () => reject("Error reading PDF file.");
            reader.readAsArrayBuffer(file);
        });
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
        .catch(error => {
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
        .catch(error => {
            alert("❌ Error uploading extracted PDF text.");
        });
    }

    function updateFileList(files) {
        fileList.innerHTML = "";
        files.forEach(file => {
            let li = document.createElement("li");
            li.className = "file-item d-flex align-items-center";
            li.innerHTML = `<i class="bi bi-file-earmark-text"></i> <span>${file.name}</span>`;
            fileList.appendChild(li);
        });
    }
});
