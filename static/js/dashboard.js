document.addEventListener("DOMContentLoaded", () => {
    modelName = document.getElementById("modelName").textContent;
    let themeColorMap = {}; 
    // console.log(modelName)
    let themes = [];
    let themeChartInstance = null;
    let themeTrendChart = null;

    // Generate visually distinctive HSL colors using the Golden Angle
function generateColors(count) {
  const colors = [];
  const goldenAngle = 137.508; // degrees

  for (let i = 0; i < count; i++) {
    const hue = (i * goldenAngle) % 360;
    const saturation = 70 + (i % 2) * 10;  // Slight variation
    const lightness = 55 + (i % 3) * 5;    // Slight variation
    colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
  }

  return colors;
}

    
    // Build Chart.js data object with meta
    function createThemeChartData(themes) {
      return {
        labels: themes.map(t => t.label),
        ids: themes.map(t => t.id),
        meta: themes.map(t => ({ id: t.id, color: t.color })), // ✅ keep color
        datasets: [{
          data: themes.map(t => t.document_count),
          backgroundColor: themes.map(t => t.color || "#0d6efd")
        }]
      };
    }
    
    
    


    // ✅ Only run this AFTER themes are fetched
    async function fetchThemes(modelName) {
      try {
        const response = await fetch("/api/themes", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ model: modelName })  // 👈 must be 'model'
        });
    
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Network error: ${response.status} - ${errorText}`);
        }
    
        let themes = await response.json();
        // console.log("Fetched themes:", themes);
    
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
      
      
fetchThemes(modelName);
    // Inject colors into each theme
const colors = generateColors(themes.length);
themes = themes.map((t, i) => ({ ...t, color: colors[i] }));

// Build chart data and render
const themeChartData = createThemeChartData(themes);
renderThemeChart(themeChartData);
    
function renderThemeChart(chartData) {
  // Save theme colors for scatter plot later
  themeColorMap = {}; // reset
  chartData.meta.forEach(meta => {
    themeColorMap[meta.id] = meta.color || "#0d6efd";
  });
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
                console.log("Clicked index:", index);
                console.log("Meta array:", chartData.meta);
            
                const meta = chartData.meta?.[index];
                if (!meta) {
                  console.error("No meta found at this index.");
                  return;
                }
            
                const themeColor = meta.color || "#0d6efd";
                loadThemeDetails(meta.id, themeColor, modelName);
                new bootstrap.Modal(document.getElementById("themeDetailModal")).show();
              }
            }
            
          
        }
      });
    }


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

  const baseColumns = ["id", "text", "theme", "score"];
  const hasRationale = docs.some(doc => "rationale" in doc);
  const columns = hasRationale
    ? ["id", "text", "theme", "rationale", "score"]
    : baseColumns;

  // Create table header
  const headerRow = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col.charAt(0).toUpperCase() + col.slice(1).replace(/_/g, " ");
    th.classList.add("medium-text");
    if (col === "id") th.style.width = "6%";
    else if (col === "score") th.style.width = "8%";
    else if (col === "rationale") th.style.width = "22%";
    else th.style.width = "auto";
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  // Create table rows
  docs.forEach(doc => {
    const row = document.createElement("tr");
    row.dataset.search = Object.values(doc).join(" ").toLowerCase();
    row.classList.add("medium-text");

    row.addEventListener("click", () => showInferenceModal(doc));

    columns.forEach(col => {
      const td = document.createElement("td");
      let value = doc[col] ?? "—";

      if (col === "id" && typeof value === "string") {
        td.textContent = value.slice(0, 4);
        td.title = value;
      } else if (col === "score" && typeof value === "number") {
        td.textContent = value.toFixed(3);
        td.title = value.toFixed(3);
      } else {
        td.textContent = value;
        td.title = value;
      }

      td.classList.add("truncate-cell", "medium-text");
      row.appendChild(td);
    });

    tbody.appendChild(row);
  });
}




  async function showInferenceModal(doc) {
    console.log(doc)
    try {
      const response = await fetch("/text-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: doc.text, id: doc.id, model: modelName })
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

    async function fetchDocuments(modelName) {
      try {
        const response = await fetch("/api/documents", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ model: modelName })
        });
    
        if (!response.ok) throw new Error("Failed to fetch documents.");
    
        const documents = await response.json();
    
        // console.log("Fetched documents:", documents);
    
        // Then render the table
        renderDocumentTable(documents);
      } catch (err) {
        console.error("Document fetch error:", err);
      }
    }
  fetchDocuments(modelName);
renderDocumentTable(documents);


function populateThemeDiagnosticsTable(themes = []) {
  const tableHead = document.querySelector("#themeStatsTable thead");
  const tableBody = document.querySelector("#themeStatsTable tbody");
  const wrapper = document.querySelector("#themeStatsTable").parentElement;

  if (!tableHead || !tableBody || !wrapper) {
    return console.error("Theme diagnostics table or wrapper elements not found");
  }



  tableHead.innerHTML = "";
  tableBody.innerHTML = "";

  if (themes.length === 0) {
    tableHead.innerHTML = `<tr><th>No Data</th></tr>`;
    tableBody.innerHTML = `<tr><td class="text-muted">No diagnostics available.</td></tr>`;
    return;
  }

  const allKeys = Object.keys(themes[0]);
  const dynamicKeys = allKeys.filter(k => k !== "theme");
  const columns = ["theme", ...dynamicKeys];

  const headerRow = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
    th.style.position = "sticky";
    th.style.top = "0";
    th.style.backgroundColor = "#fff";
    th.style.zIndex = "1";
    headerRow.appendChild(th);
  });
  tableHead.appendChild(headerRow);

  themes.forEach(theme => {
    const row = document.createElement("tr");
    columns.forEach(col => {
      let value = theme[col];
      if (typeof value === "number") {
        value = col.toLowerCase().includes("prevalence")
          ? `${(value * 100).toFixed(1)}%`
          : value.toFixed(3);
      }
      const td = document.createElement("td");
      td.title = value ?? "—";
      td.textContent = value ?? "—";
      row.appendChild(td);
    });
    tableBody.appendChild(row);
  });
}



  
  async function fetchDiagnostics(modelName) {
    try {
      const res = await fetch(`/api/diagnostics?model=${encodeURIComponent(modelName)}`);
      if (!res.ok) throw new Error("Network response was not ok");
  
      const diag = await res.json();
      populateThemeDiagnosticsTable(diag);
    } catch (error) {
      console.error("Failed to fetch diagnostics:", error);
    }
  }
  
  
  // Call the function on page load or when needed
  fetchDiagnostics(modelName);
  

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

      const bgClass = i % 2 === 0 ? "bg-light-subtle" : "bg-body-tertiary";

      col.innerHTML = `
        <div class="p-2 mb-2 ${bgClass} rounded medium-text">
          <div class="d-flex justify-content-between">
            <strong>${metric.label}</strong>
            <span class="text-muted">${typeof metric.value === "number" ? metric.value.toFixed(2) : metric.value}</span>
          </div>
        </div>
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
    thead.innerHTML = `<tr class="small"><th>No data</th></tr>`;
    tbody.innerHTML = `<tr class="small"><td class="text-muted">No documents to display.</td></tr>`;
    return;
  }

  const columns = ["id", "text", "score"];
  const hasRationale = filters.some(f => f.hasOwnProperty("rationale"));
  if (hasRationale) columns.push("rationale");

  filters.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  const headerRow = document.createElement("tr");
  headerRow.classList.add("small");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.classList.add("small");
    if (col === "id") {
      th.classList.add("id-col");
      th.style.width = "40px";
    }
    th.textContent = col.charAt(0).toUpperCase() + col.slice(1);
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  filters.forEach(filter => {
    const row = document.createElement("tr");
    row.classList.add("small");
    row.dataset.search = Object.values(filter).join(" ").toLowerCase();
    row.addEventListener("click", () => showInferenceModal?.(filter));

    columns.forEach(col => {
      const td = document.createElement("td");
      td.classList.add("small", "truncate-cell");
      let value = filter[col];

      if (col === "score" && typeof value === "number") {
        value = value.toFixed(3);
      }

      if (col === "id" && typeof value === "string") {
        value = value.slice(0, 4);
        td.classList.add("id-col");
      }

      td.title = value ?? "—";
      td.textContent = value ?? "—";
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

    async function loadThemeDetails(themeId, themeColor, modelName) {
      const res = await fetch(`/api/theme/${themeId}?model=${encodeURIComponent(modelName)}`);
      const data = await res.json();

      // Lighten the theme color for background
      const lightBackground = lightenColor(themeColor, 85);
      const modalContent = document.getElementById("themeModalContent");
      modalContent.style.backgroundColor = lightBackground;

      const modalTitle = document.getElementById("selectedThemeLabel");
      modalTitle.style.color = themeColor;
      modalTitle.style.borderLeft = `6px solid ${themeColor}`;
      modalTitle.style.paddingLeft = "0.5rem";

      document.querySelectorAll("#themeDetailModal .panel").forEach(panel => {
        panel.style.borderColor = themeColor;
        panel.style.boxShadow = `0 0 0 1px ${themeColor}`;
      });


      document.getElementById("selectedThemeLabel").textContent = data.label;
      document.getElementById("selectedThemeSummary").textContent = data.summary;

      const diagnostics = {
        "Prevalence": `${parseFloat(data.prevalence).toFixed(3)}%`,
        "Coherence": parseFloat(data.coherence).toFixed(3),
        "Keyword Uniqueness": parseFloat(data.uniqueness).toFixed(3),
        "Document Matches": parseFloat(data.theme_matches).toFixed(3)
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

      
      // ✅ Use filtered table function
      renderFilteredTable(data.documents);
      renderThemeSimilarityChart(data.similar_themes);


    }

    function renderThemeSimilarityChart(similarThemes) {
      const canvas = document.getElementById("similarityDotPlot");
      if (!canvas) return console.error("Canvas with ID 'similarityDotPlot' not found.");
      const ctx = canvas.getContext("2d");
    
      // ✅ Sort by absolute distance from 0 (more similar = closer to 0)
      const processed = similarThemes
        .map(t => ({
          theme: `Theme ${t.ID}`,
          similarity: parseFloat(t.Similarity)
        }))
        .sort((a, b) => Math.abs(a.similarity) - Math.abs(b.similarity));
    
      const labels = processed.map(t => t.theme);
      const values = processed.map(t => t.similarity);
    
      // ✅ Destroy existing chart if needed
      if (window.similarityDotPlot instanceof Chart) {
        window.similarityDotPlot.destroy();
      }
    
      // ✅ Create the chart
      window.similarityDotPlot = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "Similarity Score",
            data: values,
            backgroundColor: "#4B8DF8",
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          scales: {
            x: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Similarity Score (Closer to 0 = More Similar)"
              }
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => `Similarity: ${ctx.raw.toFixed(3)}`
              }
            }
          }
        }
      });
    }


    document.getElementById("themeDetailModal").addEventListener("hidden.bs.modal", () => {
        if (themeTrendChart instanceof Chart) {
          themeTrendChart.destroy();
          themeTrendChart = null;
        }
      });

      async function fetchThemeCoordinates(modelName) {
        try {
          const response = await fetch('/api/theme-coordinates', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: modelName })
          });
      
          if (!response.ok) throw new Error('Failed to fetch theme coordinates');
      
          const data = await response.json();
          console.log('Theme Coordinates:', data);
      
          // 👉 You can now call a function to plot, e.g.:
          plotScatterChart(data, modelName);
      
        } catch (error) {
          console.error('Error loading theme coordinates:', error);
          alert('Unable to load theme coordinates.');
        }
      }
      
      fetchThemeCoordinates(modelName);


      let scatterChartInstance = null;



      function plotScatterChart(data, modelName) {
        const ctx = document.getElementById("themeChartGrid").getContext("2d");
      
        if (scatterChartInstance) {
          scatterChartInstance.destroy();
        }
      
        const scatterPoints = data.map(d => {
          const solidColor = themeColorMap[d.id] || "rgba(66, 133, 244, 1)";
          const fillColor = solidColor.replace("rgb(", "rgba(").replace(")", ", 0.2)");
      
          return {
            x: d.x,
            y: d.y,
            label: d.label,
            id: d.id,
            color: solidColor,
            backgroundColor: fillColor,
            borderColor: solidColor
          };
        });
      
        const scatterData = {
          datasets: [
            {
              label: "Themes",
              data: scatterPoints,
              backgroundColor: scatterPoints.map(p => p.backgroundColor),
              borderColor: scatterPoints.map(p => p.borderColor),
              borderWidth: 2,
              pointRadius: 30,         // ⬅️ Bigger circles
              pointHoverRadius: 42     // ⬅️ Bigger hover effect
            }
          ]
        };
      
        const labelPlugin = {
          id: "themeLabels",
          afterDatasetsDraw(chart) {
            const { ctx } = chart;
            ctx.save();
            const meta = chart.getDatasetMeta(0);
            meta.data.forEach((point, index) => {
              const { x, y } = point.getCenterPoint();
              ctx.fillStyle = "#000";
              ctx.font = "bold 13px sans-serif";  // ⬅️ Bigger label text
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(scatterPoints[index].label, x, y);
            });
            ctx.restore();
          }
        };
      
        scatterChartInstance = new Chart(ctx, {
          type: "scatter",
          data: scatterData,
          options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
              padding: 20  // Slightly more padding inside chart
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx => ctx.raw.label
                }
              }
            },
            onClick: (e, elements) => {
              if (elements.length > 0) {
                const index = elements[0].index;
                const point = scatterPoints[index];
                const themeColor = point.color || "#0d6efd";
      
                loadThemeDetails(point.id, themeColor, modelName);
      
                const modalHeader = document.querySelector("#themeDetailModal .modal-header");
                if (modalHeader) {
                  modalHeader.style.borderBottom = `3px solid ${themeColor}`;
                }
      
                const themeLabel = document.querySelector("#themeDetailModal #modalThemeLabel");
                if (themeLabel) {
                  themeLabel.style.color = themeColor;
                }
      
                new bootstrap.Modal(document.getElementById("themeDetailModal")).show();
              }
            },
            scales: {
              x: {
                display: true,
                grid: {
                  color: "rgba(0, 0, 0, 0.1)",
                  drawTicks: true,
                  drawBorder: true
                },
                ticks: { display: false }
              },
              y: {
                display: true,
                grid: {
                  color: "rgba(0, 0, 0, 0.1)",
                  drawTicks: true,
                  drawBorder: true
                },
                ticks: { display: false }
              }
            }
          },
          plugins: [labelPlugin]
        });
      }
      
      


      
      

      
      
      



});