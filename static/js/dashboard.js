document.addEventListener("DOMContentLoaded", () => {
 // Generate N distinct colors
function generateColors(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
      const hue = Math.floor((360 / count) * i);
      colors.push(`hsl(${hue}, 70%, 60%)`);
    }
    return colors;
  }
  
  // Build chart data dynamically
  function createThemeChartData(themes) {
    return {
      labels: themes.map(t => t.label),
      datasets: [{
        data: themes.map(t => t.document_count),
        backgroundColor: generateColors(themes.length)
      }]
    };
  }
  
  // Example usage
  const themes = [
    { label: "Theme 1", document_count: 120 },
    { label: "Theme 2", document_count: 90 },
    { label: "Theme 3", document_count: 65 },
    { label: "Theme 4", document_count: 40 }
  ];
  
const themeChartData = createThemeChartData(themes);
  
renderThemeChart(themeChartData)
  function renderThemeChart(chartData) {
    const ctx = document.getElementById("themeChart").getContext("2d");
  
    new Chart(ctx, {
      type: "bar",
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false // ✅ Remove legend
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.parsed.y} documents`
            }
          }
        },
        scales: {
          x: {
            ticks: {
              maxRotation: 45,
              minRotation: 30
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Documents"
            }
          }
        },
        onClick: (e, elements) => {
          if (elements.length > 0) {
            const index = elements[0].index;
            const selectedTheme = chartData.labels[index];
            console.log("Selected theme:", selectedTheme);
            // 🔁 Optional: Trigger filter updates here
          }
        }
      }
    });
  }


  

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
  
      columns.forEach(col => {
        const td = document.createElement("td");
        const value = doc[col] ?? "—";
  
        td.classList.add("truncate-cell");
        td.title = value;
  
        // ✅ On click of any cell, show full doc details
        td.addEventListener("click", () => {
          showModalDetail({
            text: doc.text || "–",
            theme: doc.theme || "–",
            rationale: doc.rationale || "–"
          });
        });
  
        td.textContent = value;
        row.appendChild(td);
      });
  
      tbody.appendChild(row);
    });
  }
  function showModalDetail(doc) {
    document.getElementById("modalFullText").textContent = doc.text ?? "–";
    document.getElementById("modalTheme").textContent = doc.theme ?? "–";
    document.getElementById("modalRationale").textContent = doc.rationale ?? "–";
  
    const modal = new bootstrap.Modal(document.getElementById("docDetailModal"));
    modal.show();
  }
  
  
  

document.getElementById("docSearchInput").addEventListener("input", function () {
    const query = this.value.toLowerCase().trim();
    const rows = document.querySelectorAll("#docTableBody tr");
  
    rows.forEach(row => {
      const content = row.dataset.search || "";
      row.style.display = content.includes(query) ? "" : "none";
    });
  });

  const documents = [
    {
      id: "D001",
      text: "Advancements in AI are transforming healthcare diagnostics.",
      theme: "Theme 2",
      rationale: "Mentions AI use in medical diagnosis."
    },
    {
      id: "D002",
      text: "Lawmakers debate new regulations for self-driving vehicles.",
      theme: "Theme 1",
      rationale: "Focus on regulation of autonomous systems."
    },
    {
      id: "D003",
      text: "Remote work is reshaping urban infrastructure and housing trends.",
      theme: "Theme 2",
      rationale: "Connects remote work to city design."
    },
    {
      id: "D004",
      text: "Researchers discover a new biodegradable material for packaging.",
      theme: "Theme 4",
      rationale: "Eco-friendly product innovation."
    },
    {
      id: "D005",
      text: "Online learning tools see increased adoption in schools.",
      theme: "Theme 5",
      rationale: "Focus on digital education trends."
    },
    {
      id: "D006",
      text: "The biggest change was what’s happened this year, starting with George Floyd’s death and the recognition that our world has changed,” Dolan said. “For me, that raised the question of whether we should continue using a name like Indians in this new world and what lies ahead for us. That wasn’t the decision, it was merely the decision to answer the question. We went to answer the question by talking to a wide array of local and national groups. We spoke to our whole community, in one way or another. I think the answer was pretty clear that, while so many of us who have grown up with the name and thought of it as nothing more than the name of our team and that it did not intend to have a negative impact on anybody, in particular Native Americans, it was having a negative impact on those folks. Locally, the American Indian Movement of Ohio, the Lake Erie Professional Chapter of the American Indian Science and Engineering Society, the Committee of 500 Years of Dignity and Resistance and the Lake Erie Native American Council publicly advocated for a name change. Dolan said civic leaders who cater to the underserved were equally as strong in their support of a name change. A fifth-generation Clevelander, Dolan said he understands that many fans do not agree with the team’s decision. I hope that those who do not [agree with the decision] take the time, like we did, to better understand the issues and think about a role a sports team plays in the community and whether we can play that role with a name like Indians. Dolan said the team will continue to celebrate its long history as the Indians after a new name is installed. The team will continue to engage with the community during the selection process for a new name, which he acknowledged as “difficult and complex. The process of moving away from Indians is not going to be easy,” Dolan said. “We understand it’s going to be difficult for a lot of people to make that adjustment. But I hope, over time, we embrace the process of reimagining our name. Hopefully it will be a name the community can rally around, and we hope it has more than a 105-year life to it.",
      theme: "Theme 1",
      rationale: "Policy-level climate governance."
    },
    {
      id: "D007",
      text: "Wearables track real-time patient vitals more effectively.",
      theme: "Theme 2",
      rationale: "Medical technology application."
    },
    {
      id: "D008",
      text: "New blockchain infrastructure improves supply chain traceability.",
      theme: "Theme 3",
      rationale: "Tech implementation in logistics."
    },
    {
      id: "D009",
      text: "Urban farming initiatives expand in city rooftops.",
      theme: "Theme 4"
      // no rationale
    },
    {
      id: "D010",
      text: "Schools implement mental health programs for students.",
      theme: "Theme 3"
      // no rationale
    }
  ];
  
  renderDocumentTable(documents);


  function populateThemeDiagnosticsTable(themes = []) {
    const tableBody = document.querySelector("#themeStatsTable tbody");
    if (!tableBody) return console.error("Table body not found");
  
    tableBody.innerHTML = "";
  
    themes.forEach(theme => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${theme.theme}</td>
        <td>${(theme.purity_score ?? 0).toFixed(2)}</td>
        <td>${(theme.coherence ?? 0).toFixed(2)}</td>
        <td>${(theme.prevalence ?? 0).toFixed(1)}%</td>
      `;
      tableBody.appendChild(row);
    });
  }
  
  const diag = [
    {
      theme: "Theme 1",
      purity_score: 0.91,
      coherence: 0.84,
      prevalence: 22.5,
      label: "Policy & Governance"
    },
    {
      theme: "Theme 2",
      purity_score: 0.86,
      coherence: 0.78,
      prevalence: 19.1,
      label: "Health & Medicine"
    },
    {
      theme: "Theme 3",
      purity_score: 0.93,
      coherence: 0.89,
      prevalence: 16.0,
      label: "Technology Trends"
    },
    {
      theme: "Theme 4",
      purity_score: 0.81,
      coherence: 0.72,
      prevalence: 15.6,
      label: "Green Innovation"
    },

  ];
  

populateThemeDiagnosticsTable(diag);

const themeMetrics = {
    "Topic assignment confidence": {
      value: 0.87,
      note: "(range: 0 to 1)",
      id: "avgConfidence"
    },
    "Keyword overlap (avg)": {
      value: 2.3,
      id: "avgOverlap"
    },
    "Topic balance (entropy)": {
      value: 0.91,
      id: "entropyScore"
    },
    "Most unique theme": {
      value: "Theme 2 – Urban Planning",
      id: "mostUniqueTopic"
    },
    "Least confident theme": {
      value: "Theme 5 – Policy & Regulation (avg 0.61)",
      id: "leastConfTopic"
    },
    "Top emerging theme": {
      value: "Theme 8 – Generative AI",
      id: "emergingTheme"
    }
  };

  function populateThemeInsights(metrics) {
    const container = document.getElementById("themeInsightsGrid");
    container.innerHTML = ""; // Clear existing
  
    for (const [label, info] of Object.entries(metrics)) {
      const col = document.createElement("div");
      col.className = "col";
  
      col.innerHTML = `
        <strong>${label}:</strong>
        <div><span id="${info.id}" class="${info.class || ''}">${typeof info.value === "number" ? info.value.toFixed(2) : info.value}</span></div>
      `;
  
      container.appendChild(col);
    }
  }

  populateThemeInsights(themeMetrics);
  
  
  

});  