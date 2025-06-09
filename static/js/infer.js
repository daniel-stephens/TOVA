document.addEventListener("DOMContentLoaded", () => {
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

        console.log(madeInference)
          
        
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
    const rationaleDiv = document.getElementById("inferredRationaleWrapper"); // The wrapper div
    const rationaleText = document.getElementById("inferredRationale");       // The span or p for the text

    if (!rationale || rationale.trim() === "") {
      rationaleDiv.style.display = "none";
    } else {
      rationaleText.textContent = rationale;
      rationaleDiv.style.display = "block";
    }

    
    // change the text area class
    document.getElementById("textAreaSpace").className = "col-md-6";


    // Show result block
    document.getElementById("inferredResults").style.display = "block";
  
    // Draw chart
    const ctx = document.getElementById("inferredThemeChart").getContext("2d");
    if (window.inferredChart) window.inferredChart.destroy();
  
    window.inferredChart = new Chart(ctx, {
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

});
  
  