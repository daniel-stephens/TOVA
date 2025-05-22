document.addEventListener("DOMContentLoaded", () => {

    let themes = [];
    let themeChartInstance = null;
    let themeTrendChart = null;

    // Generate N distinct HSL colors
    function generateColors(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
        const hue = Math.floor((360 / count) * i);
        colors.push(`hsl(${hue}, 70%, 60%)`);
    }
    return colors;
    }
    
    // Build Chart.js data object with meta
    function createThemeChartData(themes) {
        return {
        labels: themes.map(t => t.label),
        ids: themes.map(t => t.id),
        meta: themes.map(t => ({ id: t.id, color: t.color })),
        datasets: [{
            data: themes.map(t => t.document_count),
            backgroundColor: themes.map(t => t.color || "#0d6efd")
        }]
        };
    }
    


    // ✅ Only run this AFTER themes are fetched
async function fetchThemes() {
    try {
      const response = await fetch("/api/themes");
      if (!response.ok) throw new Error("Network response was not ok");
  
      themes = await response.json();
      console.log("Fetched themes:", themes);
  
      // Inject colors here ✅
      const colors = generateColors(themes.length);
      themes = themes.map((t, i) => ({ ...t, color: colors[i] }));
  
      const chartData = createThemeChartData(themes);
      renderThemeChart(chartData);
    } catch (error) {
      console.error("Failed to fetch themes:", error);
    }
  }
    
function lightenColor(hex, percent) {
        const num = parseInt(hex.replace("#", ""), 16),
              amt = Math.round(2.55 * percent),
              R = (num >> 16) + amt,
              G = (num >> 8 & 0x00FF) + amt,
              B = (num & 0x0000FF) + amt;
        return `rgb(${Math.min(R,255)}, ${Math.min(G,255)}, ${Math.min(B,255)})`;
      }
      
      
fetchThemes();
    // Inject colors into each theme
const colors = generateColors(themes.length);
themes = themes.map((t, i) => ({ ...t, color: colors[i] }));

// Build chart data and render
const themeChartData = createThemeChartData(themes);
renderThemeChart(themeChartData);
    
function renderThemeChart(chartData) {
      const canvas = document.getElementById("themeChart");
      const ctx = canvas.getContext("2d");
    
      // Destroy old chart if exists
      if (themeChartInstance) {
        themeChartInstance.destroy();
      }
    
      // Create and save new chart
      themeChartInstance = new Chart(ctx, {
        type: "bar",
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx => `${ctx.parsed.y} documents`
                }
              }
            },
            scales: {
              x: {
                ticks: { maxRotation: 45, minRotation: 30 }
              },
              y: {
                beginAtZero: true,
                title: { display: true, text: "Documents" }
              }
            },
          
          onClick: (e, elements) => {
            if (elements.length > 0) {
              const index = elements[0].index;
              const meta = chartData.meta[index];
              const themeColor = meta.color || "#0d6efd";
          
              loadThemeDetails(meta.id, themeColor); // ✅ Pass the color
              new bootstrap.Modal(document.getElementById("themeDetailModal")).show();
            }
          }
          
        }
      });
    }



fetchThemes();
let currentDocs = []; // hold the last set of docs for filtering


function renderDocumentTable(docs = []) {
    currentDocs = docs;
  
    const thead = document.getElementById("docTableHead");
    const tbody = document.getElementById("docTableBody");
    thead.innerHTML = "";
    tbody.innerHTML = "";
  
    if (docs.length === 0) {
      thead.innerHTML = `<tr><th>No data</th></tr>`;
      tbody.innerHTML = `<tr><td class="text-muted">No documents to display.</td></tr>`;
      return;
    }
  
    const columns = Object.keys(docs[0]);
    const headerRow = document.createElement("tr");
    columns.forEach(col => {
      const th = document.createElement("th");
      th.textContent = col.charAt(0).toUpperCase() + col.slice(1).replace(/_/g, " ");
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
  
    docs.forEach(doc => {
        const row = document.createElement("tr");
        row.dataset.search = Object.values(doc).join(" ").toLowerCase();
      
        // 🔥 Attach click event to run inference + open modal
        row.addEventListener("click", () => showInferenceModal(doc));
      
        columns.forEach(col => {
          const td = document.createElement("td");
          const value = doc[col] ?? "—";
          td.classList.add("truncate-cell");
          td.title = value;
          td.textContent = value;
          row.appendChild(td);
        });
      
        tbody.appendChild(row);
      });      
      
  }

  async function showInferenceModal(doc) {
    try {
      const response = await fetch("/infer-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: doc.text })
      });
  
      const result = await response.json();

    //   console.log(result)
  
      // Populate modal
      document.getElementById("modalFullText").textContent = doc.text || "—";
      document.getElementById("modalTheme").textContent = result.theme || "—";
      document.getElementById("modalRationale").textContent = result.rationale || "—";
  
    //   console.log(result.top_themes)
      renderDocInferenceChart(result.top_themes);

  
      const modal = new bootstrap.Modal(document.getElementById("docDetailModal"), {
        backdrop: false,
        focus: true
      });
      modal.show();
      
  
    } catch (error) {
      console.error("Inference failed:", error);
      alert("Failed to infer topic.");
    }
  }


  let docInferenceChart = null; // Global variable to track the chart instance

  function renderDocInferenceChart(topThemes = []) {
    const canvas = document.getElementById("docInferenceChart");
    if (!canvas) {
      console.warn("Canvas element #docInferenceChart not found.");
      return;
    }
  
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.error("Failed to get canvas 2D context.");
      return;
    }
  
    // 🔁 Destroy the existing chart instance if it exists
    if (docInferenceChart) {
      docInferenceChart.destroy();
    }
  
    // ✅ Create and assign new chart
    docInferenceChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: topThemes.map(t => t.label),
        datasets: [{
          data: topThemes.map(t => t.score),
          backgroundColor: "#0d6efd"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 1
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `${(ctx.parsed.y * 100).toFixed(1)}%`
            }
          }
        }
      }
    });
  }
  
  

document.getElementById("docSearchInput").addEventListener("input", function () {
    const query = this.value.toLowerCase().trim();
    const rows = document.querySelectorAll("#docTableBody tr");
  
    rows.forEach(row => {
      const content = row.dataset.search || "";
      row.style.display = content.includes(query) ? "" : "none";
    });
  });

  let documents = [];

    async function fetchDocuments() {
    try {
        const response = await fetch("/api/documents");
        if (!response.ok) throw new Error("Failed to fetch documents.");
        documents = await response.json();

        console.log("Fetched documents:", documents);

        // Then render the table
        renderDocumentTable(documents);
    } catch (err) {
        console.error("Document fetch error:", err);
    }
    }

    fetchDocuments();


  
  
renderDocumentTable(documents);


function populateThemeDiagnosticsTable(themes = []) {
    const tableBody = document.querySelector("#themeStatsTable tbody");
    if (!tableBody) return console.error("Table body not found");
  
    tableBody.innerHTML = "";
  
    themes.forEach(theme => {
      const row = document.createElement("tr");
      row.classList.add("small");  // ✅ Makes row text smaller
      row.innerHTML = `
        <td>${theme.theme}</td>
        <td>${(theme.purity_score ?? 0).toFixed(2)}</td>
        <td>${(theme.coherence ?? 0).toFixed(2)}</td>
        <td>${(theme.prevalence ?? 0).toFixed(1)}%</td>
      `;
      tableBody.appendChild(row);
    });
  }
  
  async function fetchDiagnostics() {
    try {
      const res = await fetch("/api/diagnostics");
      const diag= await res.json();
      populateThemeDiagnosticsTable(diag);
    } catch (error) {
      console.error("Failed to fetch diagnostics:", error);
    }
  }
  
  // Call the function on page load or when needed
  fetchDiagnostics();
  

// populateThemeDiagnosticsTable(diag);



async function populateThemeInsightsFromAPI() {
    const container = document.getElementById("themeInsightsGrid");
    container.innerHTML = "";
  
    try {
      const res = await fetch("/api/theme-metrics");
      const data = await res.json();
      const metrics = data.metrics;
  
      metrics.forEach((metric, i) => {
        const col = document.createElement("div");
        col.className = "col-md-6";
  
        // Use light background shade based on index (alternating)
        const bgClass = i % 2 === 0 ? "bg-light-subtle" : "bg-body-tertiary";
  
        col.innerHTML = `
          <div class="p-2 mb-2 ${bgClass} small rounded">
            <div class="d-flex justify-content-between">
              <strong>${metric.label}</strong>
              <span class="text-muted">${typeof metric.value === "number" ? metric.value.toFixed(2) : metric.value}</span>
        `;
  
        container.appendChild(col);
      });
    } catch (error) {
      container.innerHTML = `<div class="text-danger">Failed to load theme insights.</div>`;
      console.error("Error loading metrics:", error);
    }
  }
  
  
populateThemeInsightsFromAPI();
    


function renderFilteredTable(filters = []) {
  currentDocs = filters;

  const thead = document.getElementById("filteredTableHead");
  const tbody = document.getElementById("filteredTableBody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  if (filters.length === 0) {
    thead.innerHTML = `<tr><th>No data</th></tr>`;
    tbody.innerHTML = `<tr><td class="text-muted">No documents to display.</td></tr>`;
    return;
  }

  const columns = ["id", "text", "rationale"];
  const headerRow = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col.charAt(0).toUpperCase() + col.slice(1);
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  filters.forEach(filter => {
    const row = document.createElement("tr");
    row.dataset.search = Object.values(filter).join(" ").toLowerCase();

    row.addEventListener("click", () => showInferenceModal?.(filter)); // Optional modal

    columns.forEach(col => {
      const td = document.createElement("td");
      const value = filter[col] ?? "—";
      td.classList.add("truncate-cell");
      td.title = value;
      td.textContent = value;
      row.appendChild(td);
    });

    tbody.appendChild(row);
  });
}

document.getElementById("filteredSearchInput").addEventListener("input", (e) => {
  const term = e.target.value.toLowerCase();
  const rows = document.querySelectorAll("#filteredTableBody tr");

  rows.forEach(row => {
    const content = row.dataset.search || "";
    row.style.display = content.includes(term) ? "" : "none";
  });
});

async function loadThemeDetails(themeId, themeColor) {
  const res = await fetch(`/api/theme/${themeId}`);
  const data = await res.json();
      // Lighten the theme color for background
  const lightBackground = lightenColor(themeColor, 85); // Light pastel background

  // Apply to modal content
  const modalContent = document.getElementById("themeModalContent");
  modalContent.style.backgroundColor = lightBackground;

  // Style modal header title
  const modalTitle = document.getElementById("selectedThemeLabel");
  modalTitle.style.color = themeColor;
  modalTitle.style.borderLeft = `6px solid ${themeColor}`;
  modalTitle.style.paddingLeft = "0.5rem";

  // Border tint on panels inside modal
  document.querySelectorAll("#themeDetailModal .panel").forEach(panel => {
    panel.style.borderColor = themeColor;
    panel.style.boxShadow = `0 0 0 1px ${themeColor}`;
  });

  document.getElementById("selectedThemeLabel").textContent = data.label;
  document.getElementById("selectedThemeSummary").textContent = data.summary;

  const diagnostics = {
    "Prevalence": `${data.prevalence}%`,
    "Coherence": data.coherence,
    "Keyword Uniqueness": data.uniqueness,
    "Document Matches": data.theme_matches
  };

  const diagContainer = document.getElementById("themeDiagnosticsGrid");
  diagContainer.innerHTML = "";
  Object.entries(diagnostics).forEach(([label, value]) => {
    const div = document.createElement("div");
    div.innerHTML = `
      <div class="border rounded p-2 bg-light-subtle h-100 small">
        <strong>${label}:</strong> <span class="text-muted">${value}</span>
      </div>
    `;
    diagContainer.appendChild(div);
  });

  const keywordContainer = document.getElementById("selectedThemeKeywords");
  keywordContainer.innerHTML = "";
  data.keywords.forEach(keyword => {
    const badge = document.createElement("span");
    badge.className = "badge bg-light text-dark border me-1 mb-1";
    badge.textContent = keyword;
    keywordContainer.appendChild(badge);
  });

  const neighborsList = document.getElementById("selectedThemeNeighbors");
  neighborsList.innerHTML = "";
  data.similar_themes.forEach(theme => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>Theme ${theme.id}:</strong> ${theme.label} (Similarity: ${theme.similarity})`;
    neighborsList.appendChild(li);
  });

  
  
   // Track the chart instance globally

   const ctx = document.getElementById("selectedThemeTrend").getContext("2d");

   // ✅ Fully destroy previous chart if it exists
   if (themeTrendChart instanceof Chart) {
     themeTrendChart.destroy();
   
     // Optional: remove Chart.js internal reference
     Chart.registry.remove(themeTrendChart);
   }
   
   themeTrendChart = new Chart(ctx, {
     type: 'line',
     data: {
       labels: data.trend_labels || ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
       datasets: [{
         label: "Document Count",
         data: data.trend,
         fill: false,
         borderColor: '#0d6efd',
         tension: 0.3
       }]
     },
     options: {
       plugins: { legend: { display: false }},
       scales: { y: { beginAtZero: true } }
     }
   });
   
      


  // ✅ Use filtered table function
  renderFilteredTable(data.documents);
}

document.getElementById("themeDetailModal").addEventListener("hidden.bs.modal", () => {
    if (themeTrendChart instanceof Chart) {
      themeTrendChart.destroy();
      themeTrendChart = null;
    }
  });

});