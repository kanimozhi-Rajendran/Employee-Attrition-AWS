/* ============================================================
   EMPLOYEE ATTRITION RISK & HR DECISION SUPPORT SYSTEM
   Frontend Controller & Chart.js Visualizations
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initGaugeChart();
  fetchAnalytics();
  fetchPresets();
  setupPredictionForm();
  setupBulkUpload();
  setupExplorerFilters();
  setupSampleCSVDownload();
});

// Global state & chart instances
let chartRiskDist = null;
let chartDept = null;
let chartOvertime = null;
let chartSatisfaction = null;
let gaugeChart = null;

let bulkProcessedData = [];
let explorerFullData = [];
let samplePresets = {};

/* ------------------------------------------------------------
   1. TAB SWITCHING LOGIC
   ------------------------------------------------------------ */
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");

      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const activePane = document.getElementById(targetId);
      if (activePane) {
        activePane.classList.add("active");
      }
    });
  });
}

/* ------------------------------------------------------------
   2. GAUGE CHART INITIALIZATION
   ------------------------------------------------------------ */
function initGaugeChart() {
  const ctx = document.getElementById("gauge-chart").getContext("2d");
  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [0.6, 99.4],
        backgroundColor: ["#10b981", "rgba(255, 255, 255, 0.08)"],
        borderWidth: 0,
        circumference: 240,
        rotation: 240,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "78%",
      plugins: {
        tooltip: { enabled: false }
      }
    }
  });
}

function updateGauge(probability, category) {
  const pct = Math.min(100, Math.max(0, probability * 100));
  let color = "#10b981"; // green
  if (category === "High") color = "#ef4444"; // red
  else if (category === "Medium") color = "#f59e0b"; // amber

  gaugeChart.data.datasets[0].data = [pct, 100 - pct];
  gaugeChart.data.datasets[0].backgroundColor = [color, "rgba(255, 255, 255, 0.08)"];
  gaugeChart.update();
}

/* ------------------------------------------------------------
   3. FETCH ANALYTICS & POPULATE CHARTS
   ------------------------------------------------------------ */
async function fetchAnalytics() {
  try {
    const res = await fetch("/analytics");
    if (!res.ok) throw new Error("Failed to fetch analytics");
    const data = await res.json();

    if (data.summary) {
      document.getElementById("kpi-total-employees").textContent = data.summary.total_employees.toLocaleString();
      document.getElementById("kpi-high-risk").textContent = data.summary.predicted_high_risk.toLocaleString();
      document.getElementById("kpi-med-risk").textContent = data.summary.predicted_medium_risk.toLocaleString();
      document.getElementById("kpi-low-risk").textContent = data.summary.predicted_low_risk.toLocaleString();
      document.getElementById("kpi-attrition-rate").textContent = `${data.summary.actual_attrition_rate}%`;
    }

    if (data.charts) {
      renderRiskDistributionChart(data.charts.risk_distribution);
      renderDepartmentChart(data.charts.department_attrition);
      renderOvertimeChart(data.charts.overtime_impact);
      renderSatisfactionChart(data.charts.job_satisfaction);
    }

    if (data.sample_employees) {
      explorerFullData = data.sample_employees;
      renderExplorerTable(explorerFullData);
    }
  } catch (err) {
    console.error("Analytics fetch error:", err);
  }
}

function renderRiskDistributionChart(chartData) {
  const ctx = document.getElementById("chart-risk-dist").getContext("2d");
  if (chartRiskDist) chartRiskDist.destroy();

  chartRiskDist = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: chartData.labels,
      datasets: [{
        data: chartData.data,
        backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
        borderColor: "rgba(11, 15, 25, 0.8)",
        borderWidth: 2,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#9ca3af", font: { family: "Inter", size: 12 } }
        }
      }
    }
  });
}

function renderDepartmentChart(chartData) {
  const ctx = document.getElementById("chart-dept-attrition").getContext("2d");
  if (chartDept) chartDept.destroy();

  chartDept = new Chart(ctx, {
    type: "bar",
    data: {
      labels: chartData.labels,
      datasets: [
        {
          label: "Stayed (Retained)",
          data: chartData.attrition_no,
          backgroundColor: "rgba(99, 102, 241, 0.7)",
          borderRadius: 4
        },
        {
          label: "Departed (Attrition)",
          data: chartData.attrition_yes,
          backgroundColor: "rgba(239, 68, 68, 0.85)",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          stacked: true,
          ticks: { color: "#9ca3af", font: { family: "Inter", size: 11 } },
          grid: { color: "rgba(255,255,255,0.04)" }
        },
        y: {
          stacked: true,
          ticks: { color: "#9ca3af", font: { family: "Inter", size: 11 } },
          grid: { color: "rgba(255,255,255,0.04)" }
        }
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#9ca3af", font: { family: "Inter", size: 12 } }
        }
      }
    }
  });
}

function renderOvertimeChart(chartData) {
  const ctx = document.getElementById("chart-overtime").getContext("2d");
  if (chartOvertime) chartOvertime.destroy();

  chartOvertime = new Chart(ctx, {
    type: "bar",
    data: {
      labels: chartData.labels,
      datasets: [
        {
          label: "Stayed",
          data: chartData.attrition_no,
          backgroundColor: "rgba(16, 185, 129, 0.7)",
          borderRadius: 4
        },
        {
          label: "Departed",
          data: chartData.attrition_yes,
          backgroundColor: "rgba(239, 68, 68, 0.85)",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: "#9ca3af", font: { family: "Inter", size: 11 } },
          grid: { color: "rgba(255,255,255,0.04)" }
        },
        y: {
          ticks: { color: "#9ca3af", font: { family: "Inter", size: 11 } },
          grid: { color: "rgba(255,255,255,0.04)" }
        }
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#9ca3af", font: { family: "Inter", size: 12 } }
        }
      }
    }
  });
}

function renderSatisfactionChart(chartData) {
  const ctx = document.getElementById("chart-satisfaction").getContext("2d");
  if (chartSatisfaction) chartSatisfaction.destroy();

  chartSatisfaction = new Chart(ctx, {
    type: "bar",
    data: {
      labels: chartData.labels,
      datasets: [
        {
          label: "Stayed",
          data: chartData.attrition_no,
          backgroundColor: "rgba(6, 182, 212, 0.7)",
          borderRadius: 4
        },
        {
          label: "Departed",
          data: chartData.attrition_yes,
          backgroundColor: "rgba(239, 68, 68, 0.85)",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: "#9ca3af", font: { family: "Inter", size: 11 } },
          grid: { color: "rgba(255,255,255,0.04)" }
        },
        y: {
          ticks: { color: "#9ca3af", font: { family: "Inter", size: 11 } },
          grid: { color: "rgba(255,255,255,0.04)" }
        }
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#9ca3af", font: { family: "Inter", size: 12 } }
        }
      }
    }
  });
}

/* ------------------------------------------------------------
   4. SINGLE PREDICTION & PRESET LOGIC
   ------------------------------------------------------------ */
async function fetchPresets() {
  try {
    const res = await fetch("/sample-presets");
    if (res.ok) {
      samplePresets = await res.json();
    }
  } catch (e) {
    console.error("Presets error:", e);
  }
}

function fillFormWithData(data) {
  document.getElementById("inp-age").value = data.Age ?? 30;
  document.getElementById("inp-gender").value = data.Gender ?? "Female";
  document.getElementById("inp-marital").value = data.MaritalStatus ?? "Single";
  document.getElementById("inp-dept").value = data.Department ?? "Research & Development";
  document.getElementById("inp-role").value = data.JobRole ?? "Research Scientist";
  document.getElementById("inp-level").value = data.JobLevel ?? 2;
  document.getElementById("inp-income").value = data.MonthlyIncome ?? 5000;
  document.getElementById("inp-stock").value = data.StockOptionLevel ?? 1;
  document.getElementById("inp-overtime").value = data.OverTime ?? "No";
  document.getElementById("inp-travel").value = data.BusinessTravel ?? "Travel_Rarely";
  document.getElementById("inp-distance").value = data.DistanceFromHome ?? 5;
  document.getElementById("inp-hike").value = data.PercentSalaryHike ?? 15;
  document.getElementById("inp-js").value = data.JobSatisfaction ?? 3;
  document.getElementById("inp-env").value = data.EnvironmentSatisfaction ?? 3;
  document.getElementById("inp-wlb").value = data.WorkLifeBalance ?? 3;
  document.getElementById("inp-involvement").value = data.JobInvolvement ?? 3;
  document.getElementById("inp-tenure").value = data.YearsAtCompany ?? 5;
  document.getElementById("inp-promo").value = data.YearsSinceLastPromotion ?? 1;
  document.getElementById("inp-mgr").value = data.YearsWithCurrManager ?? 3;
  document.getElementById("inp-companies").value = data.NumCompaniesWorked ?? 2;
  document.getElementById("inp-totalyears").value = data.TotalWorkingYears ?? 8;
}

function getFormData() {
  return {
    Age: parseInt(document.getElementById("inp-age").value),
    Gender: document.getElementById("inp-gender").value,
    MaritalStatus: document.getElementById("inp-marital").value,
    Department: document.getElementById("inp-dept").value,
    JobRole: document.getElementById("inp-role").value,
    JobLevel: parseInt(document.getElementById("inp-level").value),
    MonthlyIncome: parseInt(document.getElementById("inp-income").value),
    StockOptionLevel: parseInt(document.getElementById("inp-stock").value),
    OverTime: document.getElementById("inp-overtime").value,
    BusinessTravel: document.getElementById("inp-travel").value,
    DistanceFromHome: parseInt(document.getElementById("inp-distance").value),
    PercentSalaryHike: parseInt(document.getElementById("inp-hike").value),
    JobSatisfaction: parseInt(document.getElementById("inp-js").value),
    EnvironmentSatisfaction: parseInt(document.getElementById("inp-env").value),
    WorkLifeBalance: parseInt(document.getElementById("inp-wlb").value),
    JobInvolvement: parseInt(document.getElementById("inp-involvement").value),
    YearsAtCompany: parseInt(document.getElementById("inp-tenure").value),
    YearsSinceLastPromotion: parseInt(document.getElementById("inp-promo").value),
    YearsWithCurrManager: parseInt(document.getElementById("inp-mgr").value),
    NumCompaniesWorked: parseInt(document.getElementById("inp-companies").value),
    TotalWorkingYears: parseInt(document.getElementById("inp-totalyears").value),
    DailyRate: 800,
    HourlyRate: 60,
    MonthlyRate: 15000,
    Education: 3,
    EducationField: "Life Sciences",
    PerformanceRating: 3,
    RelationshipSatisfaction: 3,
    TrainingTimesLastYear: 3,
  };
}

function setupPredictionForm() {
  const form = document.getElementById("prediction-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await executePrediction();
  });

  // Preset buttons
  document.getElementById("btn-preset-high").addEventListener("click", () => {
    if (samplePresets.high_risk) {
      fillFormWithData(samplePresets.high_risk.data);
      executePrediction();
    }
  });

  document.getElementById("btn-preset-low").addEventListener("click", () => {
    if (samplePresets.low_risk) {
      fillFormWithData(samplePresets.low_risk.data);
      executePrediction();
    }
  });

  document.getElementById("btn-preset-mod").addEventListener("click", () => {
    if (samplePresets.moderate_risk) {
      fillFormWithData(samplePresets.moderate_risk.data);
      executePrediction();
    }
  });
}

async function executePrediction() {
  const btn = document.getElementById("btn-predict-submit");
  const originalText = btn.innerHTML;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Employee Profile...';
  btn.disabled = true;

  try {
    const payload = getFormData();
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed");
    }

    const result = await res.json();
    renderPredictionResult(result);
  } catch (err) {
    alert("Prediction Error: " + err.message);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

function renderPredictionResult(result) {
  // Update badge
  const badge = document.getElementById("res-badge");
  badge.className = `badge ${result.risk_badge_class}`;
  badge.textContent = `${result.risk_category} Risk`;

  // Update Percentage & Gauge
  document.getElementById("res-percentage").textContent = result.attrition_risk_percentage;
  updateGauge(result.attrition_risk_score, result.risk_category);

  // Update Headline & Wording
  const headline = document.getElementById("res-headline");
  if (result.risk_category === "High") {
    headline.textContent = "Predicted: Elevated Attrition Risk";
  } else if (result.risk_category === "Medium") {
    headline.textContent = "Predicted: Moderate Attrition Risk";
  } else {
    headline.textContent = "Predicted: Stable Retention";
  }
  document.getElementById("res-wording").textContent = result.summary_wording;

  // Render Top Risk Factors
  const riskList = document.getElementById("list-risk-factors");
  riskList.innerHTML = "";
  if (result.top_risk_factors && result.top_risk_factors.length > 0) {
    result.top_risk_factors.forEach(f => {
      const el = document.createElement("div");
      el.className = "factor-item";
      el.innerHTML = `
        <div class="factor-header">
          <span style="color: #fff;">${f.feature_label}</span>
          <span style="color: #f87171; font-size: 0.75rem;">+${f.importance_score} Impact</span>
        </div>
        <div class="factor-desc">${f.description}</div>
        <div class="factor-bar-bg">
          <div class="factor-bar-fill factor-bar-risk" style="width: ${Math.min(100, f.importance_score * 80)}%;"></div>
        </div>
      `;
      riskList.appendChild(el);
    });
  } else {
    riskList.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-muted);">No major elevated risk drivers identified for this profile.</div>';
  }

  // Render Top Protective Factors
  const protectList = document.getElementById("list-protective-factors");
  protectList.innerHTML = "";
  if (result.top_protective_factors && result.top_protective_factors.length > 0) {
    result.top_protective_factors.forEach(f => {
      const el = document.createElement("div");
      el.className = "factor-item";
      el.innerHTML = `
        <div class="factor-header">
          <span style="color: #fff;">${f.feature_label}</span>
          <span style="color: #4ade80; font-size: 0.75rem;">-${f.importance_score} Protective</span>
        </div>
        <div class="factor-desc">${f.description}</div>
        <div class="factor-bar-bg">
          <div class="factor-bar-fill factor-bar-protect" style="width: ${Math.min(100, f.importance_score * 80)}%;"></div>
        </div>
      `;
      protectList.appendChild(el);
    });
  } else {
    protectList.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-muted);">Standard baseline retention characteristics.</div>';
  }

  // Render HR Recommendations
  const recList = document.getElementById("list-recommendations");
  recList.innerHTML = "";
  if (result.hr_recommendations && result.hr_recommendations.length > 0) {
    result.hr_recommendations.forEach(r => {
      const li = document.createElement("li");
      li.className = "recommendation-item";
      li.innerHTML = `<i class="fa-solid fa-check" style="margin-right: 0.4rem; color: var(--accent-cyan);"></i> ${r}`;
      recList.appendChild(li);
    });
  }
}

/* ------------------------------------------------------------
   5. CSV BULK UPLOAD & PROCESSING
   ------------------------------------------------------------ */
function setupBulkUpload() {
  const dropzone = document.getElementById("bulk-dropzone");
  const fileInput = document.getElementById("bulk-file-input");

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleBulkFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleBulkFileUpload(fileInput.files[0]);
    }
  });

  document.getElementById("btn-export-bulk-csv").addEventListener("click", exportBulkResultsCSV);
}

async function handleBulkFileUpload(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Please select a valid .csv file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const dropzone = document.getElementById("bulk-dropzone");
  const originalHTML = dropzone.innerHTML;
  dropzone.innerHTML = `
    <div class="upload-icon"><i class="fa-solid fa-spinner fa-spin"></i></div>
    <h3 style="color: #fff;">Processing ${file.name}...</h3>
    <p style="color: var(--text-secondary); font-size: 0.85rem;">Applying ML pipeline and risk scoring</p>
  `;

  try {
    const res = await fetch("/predict-bulk", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Batch processing failed.");
    }

    const bulkResp = await res.json();
    bulkProcessedData = bulkResp.results || [];

    // Show metrics
    document.getElementById("bulk-summary-container").style.display = "block";
    document.getElementById("bulk-kpi-total").textContent = bulkResp.total_processed.toLocaleString();
    document.getElementById("bulk-kpi-high").textContent = bulkResp.high_risk_count.toLocaleString();
    document.getElementById("bulk-kpi-med").textContent = bulkResp.medium_risk_count.toLocaleString();
    document.getElementById("bulk-kpi-low").textContent = bulkResp.low_risk_count.toLocaleString();

    // Render table
    renderBulkTable(bulkProcessedData);
    document.getElementById("bulk-results-section").style.display = "block";
  } catch (err) {
    alert("Bulk Upload Error: " + err.message);
  } finally {
    dropzone.innerHTML = originalHTML;
  }
}

function renderBulkTable(items) {
  const tbody = document.getElementById("bulk-table-body");
  tbody.innerHTML = "";

  items.slice(0, 100).forEach(item => {
    const tr = document.createElement("tr");
    let badgeClass = "badge-low";
    if (item.risk_category === "High") badgeClass = "badge-high";
    else if (item.risk_category === "Medium") badgeClass = "badge-medium";

    tr.innerHTML = `
      <td style="font-weight: 600;">#${item.employee_id}</td>
      <td>${item.department}</td>
      <td>${item.job_role}</td>
      <td><strong>${item.prediction}</strong></td>
      <td>${item.risk_percentage}</td>
      <td><span class="badge ${badgeClass}">${item.risk_category}</span></td>
      <td style="font-size: 0.78rem; color: var(--text-secondary);">${item.top_factors_summary}</td>
    `;
    tbody.appendChild(tr);
  });
}

function exportBulkResultsCSV() {
  if (!bulkProcessedData || bulkProcessedData.length === 0) {
    alert("No batch prediction results to export.");
    return;
  }

  const headers = ["Employee_ID", "Department", "JobRole", "Attrition_Prediction", "Risk_Probability", "Risk_Percentage", "Risk_Category", "Key_Factors"];
  const rows = bulkProcessedData.map(d => [
    d.employee_id,
    `"${d.department}"`,
    `"${d.job_role}"`,
    d.prediction,
    d.risk_probability,
    d.risk_percentage,
    d.risk_category,
    `"${d.top_factors_summary}"`
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `attrition_predictions_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function setupSampleCSVDownload() {
  document.getElementById("btn-download-sample-csv").addEventListener("click", () => {
    const headers = [
      "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
      "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate",
      "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
      "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime", "PercentSalaryHike",
      "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
      "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
      "YearsSinceLastPromotion", "YearsWithCurrManager"
    ];
    const sampleRow = [
      32, "Travel_Rarely", 800, "Research & Development", 6,
      3, "Life Sciences", 3, "Female", 60,
      3, 2, "Research Scientist", 3, "Single",
      5200, 15000, 2, "Yes", 14,
      3, 3, 1, 8,
      3, 3, 5, 3,
      1, 3
    ];
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), sampleRow.join(",")].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "employee_sample_template.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
}

/* ------------------------------------------------------------
   6. EXPLORER TABLE & SEARCH/FILTER
   ------------------------------------------------------------ */
function setupExplorerFilters() {
  const searchInput = document.getElementById("explorer-search");
  const deptSelect = document.getElementById("explorer-filter-dept");
  const riskSelect = document.getElementById("explorer-filter-risk");

  function applyFilters() {
    const query = searchInput.value.toLowerCase().trim();
    const dept = deptSelect.value;
    const risk = riskSelect.value;

    const filtered = explorerFullData.filter(emp => {
      const matchQuery = !query ||
        String(emp.EmployeeNumber || "").includes(query) ||
        String(emp.JobRole || "").toLowerCase().includes(query) ||
        String(emp.Department || "").toLowerCase().includes(query);

      const matchDept = (dept === "All") || (emp.Department === dept);
      
      let empRisk = "Low";
      if (emp.Predicted_Risk_Probability >= 0.6 || emp.Attrition === "Yes") empRisk = "High";
      else if (emp.Predicted_Risk_Probability >= 0.3) empRisk = "Medium";

      const matchRisk = (risk === "All") || (empRisk === risk);

      return matchQuery && matchDept && matchRisk;
    });

    renderExplorerTable(filtered);
  }

  searchInput.addEventListener("input", applyFilters);
  deptSelect.addEventListener("change", applyFilters);
  riskSelect.addEventListener("change", applyFilters);
}

function renderExplorerTable(records) {
  const tbody = document.getElementById("explorer-table-body");
  tbody.innerHTML = "";

  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No matching employee records found.</td></tr>';
    return;
  }

  records.slice(0, 50).forEach(r => {
    const tr = document.createElement("tr");
    
    const isAttrition = (r.Attrition === "Yes");
    const proba = r.Predicted_Risk_Probability !== undefined ? r.Predicted_Risk_Probability : (isAttrition ? 0.78 : 0.08);
    
    let riskCat = "Low";
    let badgeClass = "badge-low";
    if (proba >= 0.6) { riskCat = "High"; badgeClass = "badge-high"; }
    else if (proba >= 0.3) { riskCat = "Medium"; badgeClass = "badge-medium"; }

    tr.innerHTML = `
      <td>#${r.EmployeeNumber || r.EmployeeId || 1000}</td>
      <td>${r.Age || 30}</td>
      <td>${r.Department || "R&D"}</td>
      <td>${r.JobRole || "Scientist"}</td>
      <td>$${Number(r.MonthlyIncome || 5000).toLocaleString()}</td>
      <td><span style="color: ${r.OverTime === 'Yes' ? '#f87171' : '#9ca3af'}">${r.OverTime || 'No'}</span></td>
      <td>${r.JobSatisfaction || 3}/4</td>
      <td>${r.WorkLifeBalance || 3}/4</td>
      <td>${(proba * 100).toFixed(1)}%</td>
      <td><span class="badge ${badgeClass}">${riskCat}</span></td>
    `;
    tbody.appendChild(tr);
  });
}
