(() => {
  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzone-empty");
  const previewImage = document.getElementById("preview-image");
  const analyzeBtn = document.getElementById("analyze-btn");
  const analyzeBtnLabel = document.getElementById("analyze-btn-label");
  const uploadStatus = document.getElementById("upload-status");
  const errorBanner = document.getElementById("error-banner");
  const errorText = document.getElementById("error-text");
  const results = document.getElementById("results");

  const historyEmpty = document.getElementById("history-empty");
  const historyTableWrap = document.getElementById("history-table-wrap");
  const historyTbody = document.getElementById("history-tbody");
  const historyRefreshBtn = document.getElementById("history-refresh-btn");

  // The file (and its identity) currently loaded into the picker.
  // `analyzedFileKey` records which file the results on screen belong
  // to — if the user picks a different file, results are hidden again
  // until they click Analyze, so a new upload can never be shown next
  // to a stale, previous analysis.
  let selectedFile = null;
  let analyzedFileKey = null;

  function fileKey(file) {
    return `${file.name}:${file.size}:${file.lastModified}`;
  }

  function resetResults() {
    results.classList.add("hidden");
    errorBanner.classList.add("hidden");
  }

  function handleFileSelected(file) {
    if (!file) return;

    const validTypes = ["image/png", "image/jpeg", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      uploadStatus.textContent = "Please choose a PNG or JPG image.";
      return;
    }

    selectedFile = file;
    analyzeBtn.disabled = false;
    uploadStatus.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.classList.remove("hidden");
      dropzoneEmpty.classList.add("hidden");
    };
    reader.readAsDataURL(file);

    // A newly selected image immediately invalidates any results
    // currently on screen — they belonged to whatever was analyzed
    // before, not to this new file.
    if (analyzedFileKey !== fileKey(file)) {
      resetResults();
    }
  }

  fileInput.addEventListener("change", (e) => {
    handleFileSelected(e.target.files[0]);
  });

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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading || !selectedFile;
    analyzeBtn.classList.toggle("loading", isLoading);
    analyzeBtnLabel.textContent = isLoading
      ? "Analyzing..."
      : "🔍 Analyze Prescription";
  }

  function showError(message) {
    errorText.textContent = message;
    errorBanner.classList.remove("hidden");
    results.classList.add("hidden");
  }

  function renderExplanation(explanation) {
    const container = document.getElementById("explanation-list");
    container.innerHTML = "";

    if (!explanation || explanation.length === 0) {
      const empty = document.createElement("p");
      empty.className = "status-text";
      empty.textContent = "Explanation unavailable for this prediction.";
      container.appendChild(empty);
      return;
    }

    const maxAbs = Math.max(...explanation.map((item) => Math.abs(item.contribution)));

    explanation.forEach((item) => {
      const row = document.createElement("div");
      row.className = "explanation-row";

      const label = document.createElement("div");
      label.className = "explanation-label";
      label.textContent = `${item.feature} (${item.value})`;

      const barTrack = document.createElement("div");
      barTrack.className = "explanation-bar-track";
      const bar = document.createElement("div");
      bar.className = `explanation-bar ${item.direction === "increases_risk" ? "risk" : "safe"}`;
      const pct = maxAbs > 0 ? (Math.abs(item.contribution) / maxAbs) * 100 : 0;
      bar.style.width = `${pct}%`;
      barTrack.appendChild(bar);

      const value = document.createElement("div");
      value.className = "explanation-value";
      value.textContent = (item.contribution > 0 ? "+" : "") + item.contribution.toFixed(3);

      row.appendChild(label);
      row.appendChild(barTrack);
      row.appendChild(value);
      container.appendChild(row);
    });
  }

  function renderResults(data) {
    document.getElementById("patient-name").textContent = data.patient.name;
    document.getElementById("patient-age").textContent = data.patient.age ?? "—";
    document.getElementById("patient-gender").textContent = data.patient.gender;
    document.getElementById("doctor-name").textContent = data.doctor.name;
    document.getElementById("doctor-department").textContent = data.doctor.department;

    document.getElementById("ocr-text").textContent = data.ocr_text;

    document.getElementById("metric-age").textContent = data.metrics.age;
    document.getElementById("metric-medcount").textContent = data.metrics.medicine_count;
    document.getElementById("metric-toxicity").textContent = data.metrics.toxicity_total;

    document.getElementById("confidence-value").textContent = `${data.confidence}%`;
    document.getElementById("confidence-bar").style.width = `${data.confidence}%`;

    document.getElementById("risk-value").textContent = `${data.risk_score}/100`;
    document.getElementById("risk-bar").style.width = `${data.risk_score}%`;

    const riskFactorsEl = document.getElementById("risk-factors");
    riskFactorsEl.innerHTML = "";
    if (data.risk_factors.length === 0) {
      const chip = document.createElement("span");
      chip.className = "chip ok";
      chip.textContent = "✅ No major risk factors detected";
      riskFactorsEl.appendChild(chip);
    } else {
      data.risk_factors.forEach((factor) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = `⚠ ${factor}`;
        riskFactorsEl.appendChild(chip);
      });
    }

    const verdictBanner = document.getElementById("verdict-banner");
    const verdictIcon = document.getElementById("verdict-icon");
    const verdictTitle = document.getElementById("verdict-title");
    const verdictSubtitle = document.getElementById("verdict-subtitle");

    if (data.unknown_medicine_present) {
      verdictBanner.classList.add("danger");
      verdictIcon.textContent = "🚨";
      verdictTitle.textContent = "MEDICINE NOT FOUND";
      verdictSubtitle.textContent =
        "Unrecognized medicine(s) — " +
        (data.unknown_medicine_names || []).join(", ") +
        " — manual verification required before dispensing.";
    } else if (data.requires_verification) {
      verdictBanner.classList.add("danger");
      verdictIcon.textContent = "🚨";
      verdictTitle.textContent = "VERIFICATION REQUIRED";
      verdictSubtitle.textContent = "Doctor/Hospital verification required before dispensing.";
    } else {
      verdictBanner.classList.remove("danger");
      verdictIcon.textContent = "✅";
      verdictTitle.textContent = "SAFE TO DISPENSE";
      verdictSubtitle.textContent = "Medicine can be dispensed safely.";
    }

    renderExplanation(data.explanation);

    const table = document.getElementById("feature-table");
    table.innerHTML = "";
    const headerRow = document.createElement("tr");
    const valueRow = document.createElement("tr");
    Object.entries(data.feature_data).forEach(([key, value]) => {
      const th = document.createElement("th");
      th.textContent = key;
      headerRow.appendChild(th);

      const td = document.createElement("td");
      td.textContent = value;
      valueRow.appendChild(td);
    });
    table.appendChild(headerRow);
    table.appendChild(valueRow);

    errorBanner.classList.add("hidden");
    results.classList.remove("hidden");
  }

  // ===================== HISTORY =====================

  function renderHistory(records) {
    historyTbody.innerHTML = "";

    if (!records || records.length === 0) {
      historyEmpty.classList.remove("hidden");
      historyTableWrap.classList.add("hidden");
      return;
    }

    historyEmpty.classList.add("hidden");
    historyTableWrap.classList.remove("hidden");

    records.forEach((record) => {
      const tr = document.createElement("tr");
      [
        record.prescription_id,
        record.patient_name || "—",
        record.patient_age ?? "—",
        record.patient_gender || "—",
        record.doctor_name || "—",
        record.upload_date,
        record.upload_time,
      ].forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      historyTbody.appendChild(tr);
    });
  }

  async function loadHistory() {
    try {
      const response = await fetch("/api/history");
      if (!response.ok) return;
      renderHistory(await response.json());
    } catch (_) {
      // History is a supplementary view — a failed fetch here shouldn't
      // block the upload/analyze flow from working.
    }
  }

  historyRefreshBtn.addEventListener("click", loadHistory);

  loadHistory();

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    setLoading(true);
    resetResults();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = `Request failed with status ${response.status}`;
        try {
          const errJson = await response.json();
          detail = errJson.detail || detail;
        } catch (_) {
          /* response body wasn't JSON */
        }
        throw new Error(detail);
      }

      const data = await response.json();
      analyzedFileKey = fileKey(selectedFile);
      renderResults(data);
      loadHistory();
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  });
})();
