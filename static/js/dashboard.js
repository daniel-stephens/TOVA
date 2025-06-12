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

// console.log(themeChartData)
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
      
      const rationale = result.rationale?.trim();
      const rationaleDiv = document.getElementById("rationalDiv");

      if (rationale) {
        // If div doesn't exist yet, create it
        if (!rationaleDiv) {
          const newDiv = document.createElement("div");
          newDiv.id = "rationalDiv";
          newDiv.className = "mb-3";

          const strong = document.createElement("strong");
          strong.textContent = "Rationale:";

          const p = document.createElement("p");
          p.id = "modalRationale";
          p.className = "fst-italic small text-muted mb-0";
          p.style.whiteSpace = "pre-wrap";
          p.textContent = rationale;

          newDiv.appendChild(strong);
          newDiv.appendChild(p);

          document.getElementById("inferenceOutput").appendChild(newDiv);
        } else {
          // If already exists, just update the text
          document.getElementById("modalRationale").textContent = rationale;
        }
      } else {
        // Remove it only if it exists
        if (rationaleDiv) {
          rationaleDiv.remove();
        }
      }




    //   console.log(result.top_themes)
      
      console.log(result.top_themes)
      renderDocInferenceChart(result.top_themes, modelName);

  
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

  function renderDocInferenceChart(topThemes = [], modelName = "") {
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
  
    // 🎨 Safe color lookup function
    const getThemeColor = (id) =>
      themeColorMap["t" + id] || themeColorMap[id] || "#0d6efd";
  
    const backgroundColors = topThemes.map(t => getThemeColor(t.theme_id));
  
    // ✅ Create and assign new chart
    docInferenceChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: topThemes.map(t => t.label),
        datasets: [{
          data: topThemes.map(t => t.score),
          backgroundColor: backgroundColors
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
        },
        onClick: (e, elements) => {
          if (elements.length > 0) {
            const index = elements[0].index;
            const theme = topThemes[index];
            const themeId = theme.theme_id.startsWith("t") ? theme.theme_id : "t" + theme.theme_id;
            const color = getThemeColor(theme.theme_id);
  
            loadThemeDetails(themeId, color, modelName);
  
            // Optional: update modal styling
            const modalHeader = document.querySelector("#themeDetailModal .modal-header");
            if (modalHeader) {
              modalHeader.style.borderBottom = `3px solid ${color}`;
            }
  
            const themeLabel = document.querySelector("#themeDetailModal #modalThemeLabel");
            if (themeLabel) {
              themeLabel.style.color = color;
            }
  
            new bootstrap.Modal(document.getElementById("themeDetailModal")).show();
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

    metrics.forEach((metric) => {
      const col = document.createElement("div");
      col.className = "col-md-6";

      col.innerHTML = `
        <div class="d-flex justify-content-between align-items-center p-3 rounded bg-light shadow-sm">
          <div class="text-muted small fw-semibold">${metric.label}</div>
          <div class="fw-bold text-dark small">
            ${typeof metric.value === "number" ? metric.value.toFixed(2) : metric.value}
          </div>
        </div>
      `;

      container.appendChild(col);
    });
  } catch (error) {
    container.innerHTML = `<div class="text-danger small">⚠️ Failed to load theme insights.</div>`;
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
      renderThemeSimilarityChart(data.similar_themes, modelName);


    }

    function renderThemeSimilarityChart(similarThemes, modelName = "") {
      const canvas = document.getElementById("similarityDotPlot");
      if (!canvas) return console.error("Canvas with ID 'similarityDotPlot' not found.");
      const ctx = canvas.getContext("2d");
    
      // ✅ Sort by absolute distance from 0
      const processed = similarThemes
        .map(t => ({
          id: t.ID,
          theme: `Theme ${t.ID}`,
          similarity: parseFloat(t.Similarity)
        }))
        .sort((a, b) => Math.abs(a.similarity) - Math.abs(b.similarity));
    
      const labels = processed.map(t => t.theme);
      const values = processed.map(t => t.similarity);
      const backgroundColors = processed.map(t => themeColorMap["t" + t.id] || "#4B8DF8");
    
      if (window.similarityDotPlot instanceof Chart) {
        window.similarityDotPlot.destroy();
      }
    
      window.similarityDotPlot = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "Similarity Score",
            data: values,
            backgroundColor: backgroundColors,
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
          },
          onClick: (e, elements) => {
            if (elements.length > 0) {
              const index = elements[0].index;
              const theme = processed[index];
              const themeId = "t" + theme.id;
              const color = themeColorMap[themeId] || "#0d6efd";
    
              loadThemeDetails(themeId, color, modelName);
    
              const modalHeader = document.querySelector("#themeDetailModal .modal-header");
              if (modalHeader) {
                modalHeader.style.borderBottom = `3px solid ${color}`;
              }
    
              const themeLabel = document.querySelector("#themeDetailModal #modalThemeLabel");
              if (themeLabel) {
                themeLabel.style.color = color;
              }
    
              new bootstrap.Modal(document.getElementById("themeDetailModal")).show();
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
          // console.log('Theme Coordinates:', data);
          
          // console.log(data)
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
      
        const minSize = Math.min(...data.map(d => d.size));
        const maxSize = Math.max(...data.map(d => d.size));
      
        // Normalize size to a range (e.g., 10 to 40)
        const normalizeSize = (s) => {
          const minRadius = 10;
          const maxRadius = 40;
          if (maxSize === minSize) return (minRadius + maxRadius) / 2;
          return ((s - minSize) / (maxSize - minSize)) * (maxRadius - minRadius) + minRadius;
        };
      
        if (scatterChartInstance) {
          scatterChartInstance.destroy();
        }
          const scatterPoints = data.map(d => {
          const solidColor = themeColorMap[d.id] || "rgba(66, 133, 244, 1)";
          const fillColor = solidColor.replace("rgb(", "rgba(").replace(")", ", 0.2)");
      
          return {
            x: d.x,
            y: d.y,
            r: normalizeSize(d.size),  // 👈 Used dynamically below
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
              borderWidth: 2
              // ⛔ Remove fixed pointRadius here — use dynamic one in options
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
              ctx.font = "bold 13px sans-serif";
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
              padding: 20
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx => ctx.raw.label
                }
              }
            },
            elements: {
              point: {
                radius: ctx => {
                  const index = ctx.dataIndex;
                  return scatterPoints[index].r || 20;
                },
                hoverRadius: ctx => {
                  const index = ctx.dataIndex;
                  return (scatterPoints[index].r || 20) * 1.2;
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
      
      
      
      const toggleMetricBtn = document.getElementById("toggleMetricBtn");
      const themeMetricsPanel = document.getElementById("themeMetricsPanel");
      const modelMetricsPanel = document.getElementById("modelMetricsPanel");
      
      toggleMetricBtn.addEventListener("click", () => {
        const showingTheme = !themeMetricsPanel.classList.contains("d-none");
      
        themeMetricsPanel.classList.toggle("d-none", showingTheme);
        modelMetricsPanel.classList.toggle("d-none", !showingTheme);
      
        toggleMetricBtn.textContent = showingTheme
          ? "Switch to Theme Metrics"
          : "Switch to Model Metrics";
      });

      
    /// Inference
    const toggleTextBtn = document.getElementById("toggleTextInput");
    const toggleFileBtn = document.getElementById("toggleFileInput");
    const textArea = document.getElementById("textAreaSpace");
    const fileGroup = document.getElementById("fileInputGroup");
    const inferedBlock = document.getElementById("inferredResults");
    let madeInference = false;
    toggleTextBtn.addEventListener("click", () => {
          textArea.style.display = "block";
          fileGroup.style.display = "none";
          toggleTextBtn.classList.add("active");
          toggleFileBtn.classList.remove("active");
          if (madeInference) {
              inferedBlock.style.display = "block";
            }
  
          // console.log(madeInference)
            
          
        });
        
        toggleFileBtn.addEventListener("click", () => {
          textArea.style.display = "none";
          fileGroup.style.display = "block";
          toggleFileBtn.classList.add("active");
          toggleTextBtn.classList.remove("active");
          inferedBlock.style.display = "none";
        });
  
        function renderInferenceResults(topThemes = [], rationale = "") {
          if (!topThemes || topThemes.length === 0) return;
        
          // Theme + rationale
          document.getElementById("inferredTheme").textContent = topThemes[0].label || "–";
          const rationaleDiv = document.getElementById("inferredRationaleWrapper");
          const rationaleText = document.getElementById("inferredRationale");
        
          if (!rationale || rationale.trim() === "") {
            rationaleDiv.style.display = "none";
          } else {
            rationaleText.textContent = rationale;
            rationaleDiv.style.display = "block";
          }
        
          // Adjust layout
          document.getElementById("textAreaSpace").className = "col-md-6";
        
          // Show results block
          document.getElementById("inferredResults").style.display = "block";
        
          // Get canvas context
          const ctx = document.getElementById("inferredThemeChart").getContext("2d");
          if (window.inferredChart) window.inferredChart.destroy();
        
          // Get background colors from themeColorMap

          
          const backgroundColors = topThemes.map(t => themeColorMap[t.theme_id] || "#0d6efd");
        
          // Create chart
          window.inferredChart = new Chart(ctx, {
            type: "bar",
            data: {
              labels: topThemes.map(t => t.label),
              datasets: [{
                data: topThemes.map(t => t.score),
                backgroundColor: backgroundColors
              }]
            },
            options: {
              responsive: true,
              scales: {
                y: { beginAtZero: true, max: 1 }
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
        

  // console.log(modelName)
    document.getElementById("inferBtn").addEventListener("click", async () => {
      const text = document.getElementById("inputText").value.trim();
      madeInference = true;
      if (!text) return alert("Please enter some text.");
    
      try {
        const response = await fetch("/infer-text", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ text: text, id: "id", model: modelName })
        });
    
        if (!response.ok) throw new Error("Failed to fetch inference");
    
        const result = await response.json();
        const { top_themes, rationale } = result;
    
        renderInferenceResults(top_themes, rationale);
      } catch (err) {
        console.error("Inference error:", err);
        alert("Failed to perform inference.");
      }
    }); 

      
      

      
      console.log(themeColorMap)
      



});