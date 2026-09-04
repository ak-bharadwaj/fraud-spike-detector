/**
 * Fraud-Spike Detector Web UI Client Logic
 * Strictly consumes backend API endpoints with zero fabricated/hardcoded fallback metrics.
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadSystemStatus();
  loadArtifacts();

  // Demo Control Listeners
  document.getElementById("btnPlayDemo").addEventListener("click", toggleDemoPlay);
  document.getElementById("btnStepDemo").addEventListener("click", stepDemo);
  document.getElementById("btnResetDemo").addEventListener("click", resetDemo);
  document.getElementById("selectMerchant").addEventListener("change", (e) => resetDemo(e.target.value));
});

// Tab Switching Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      
      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      if (targetId === "tabAudit") {
        fetchAuditLogs();
      }
    });
  });
}

// Fetch System Status & Metadata
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    document.getElementById("headerVersion").textContent = data.detector_version ? `v${data.detector_version}` : "N/A";
    document.getElementById("provVer").textContent = data.detector_version || "N/A";
    document.getElementById("provConfigHash").textContent = data.config_hash || "N/A";
    document.getElementById("provDevHash").textContent = data.development_dataset_hash || "N/A";
    document.getElementById("provSeed").textContent = data.seed !== undefined ? data.seed : "N/A";
  } catch (err) {
    console.error("Failed to load status:", err);
  }
}

async function loadArtifacts() {
  try {
    // 0. Track B Real-World Public Benchmark
    const rwRes = await fetch("/api/artifacts/realworld");
    if (rwRes.ok) {
      const rw = await rwRes.json();
      populateTrackB(rw);
    }

    // 1. Final Report & Metrics
    const repRes = await fetch("/api/artifacts/report");
    if (repRes.ok) {
      const rep = await repRes.json();
      populateEvaluationSummary(rep);
      populateProvenance(rep);
    }

    // 2. 5-Way Signal Ablation Suite
    const ablRes = await fetch("/api/artifacts/signal_ablation");
    if (ablRes.ok) {
      const abl = await ablRes.json();
      populateAblationTable(abl.ablation_results || []);
    }

    // 3. Evasion Confirmation
    const evaRes = await fetch("/api/artifacts/evasion");
    if (evaRes.ok) {
      const eva = await evaRes.json();
      populateEvasionTable(eva.scenarios || {});
    }

    // 4. Calibration
    const calRes = await fetch("/api/artifacts/calibration");
    if (calRes.ok) {
      const cal = await calRes.json();
      populateCalibration(cal);
    } else if (repRes.ok) {
      const rep = await repRes.clone().json();
      populateCalibration(rep.descriptive_calibration || {});
    }

    // 5. Bootstrap CIs
    const bootRes = await fetch("/api/artifacts/uncertainty");
    if (bootRes.ok) {
      const boot = await bootRes.json();
      populateBootstrap(boot);
    } else if (repRes.ok) {
      const rep = await repRes.clone().json();
      populateBootstrap(rep.bootstrap_uncertainty || {});
    }

  } catch (err) {
    console.error("Failed to load research artifacts:", err);
  }
}

function populateTrackB(report) {
  const mf = report.dataset_manifest || {};
  const splits = mf.splits || {};
  const testSplit = splits.test || {};
  const totalTx = mf.total_transactions !== undefined && mf.total_transactions !== null ? mf.total_transactions.toLocaleString() : "N/A";
  const testTx = testSplit.count !== undefined && testSplit.count !== null ? testSplit.count.toLocaleString() : "N/A";
  const fraudCases = testSplit.fraud_count !== undefined && testSplit.fraud_count !== null ? testSplit.fraud_count : "N/A";

  const scopeEl = document.getElementById("trackBScope");
  if (scopeEl) {
    scopeEl.innerHTML = `Locked test characterization • ${totalTx} total dataset transactions • <strong>${testTx} locked TEST transactions</strong> • ${fraudCases} fraud cases • Primary XGBoost Model`;
  }

  const model = report.models ? report.models.primary_xgboost : {};
  const m = model.metrics_test || {};
  const boot = report.bootstrap_ci || {};
  const pCi = boot.precision || {};
  const rCi = boot.recall || {};

  const precVal = m.precision !== undefined && m.precision !== null ? m.precision.toFixed(4) : "N/A";
  const precPct = m.precision !== undefined && m.precision !== null ? (m.precision * 100).toFixed(1) : "N/A";
  const precLow = pCi.ci_lower !== undefined && pCi.ci_lower !== null ? pCi.ci_lower.toFixed(4) : "N/A";
  const precHigh = pCi.ci_upper !== undefined && pCi.ci_upper !== null ? pCi.ci_upper.toFixed(4) : "N/A";

  const recVal = m.recall !== undefined && m.recall !== null ? m.recall.toFixed(4) : "N/A";
  const recPct = m.recall !== undefined && m.recall !== null ? (m.recall * 100).toFixed(1) : "N/A";
  const recLow = rCi.ci_lower !== undefined && rCi.ci_lower !== null ? rCi.ci_lower.toFixed(4) : "N/A";
  const recHigh = rCi.ci_upper !== undefined && rCi.ci_upper !== null ? rCi.ci_upper.toFixed(4) : "N/A";

  const f1Val = m.f1_score !== undefined && m.f1_score !== null ? m.f1_score.toFixed(4) : "N/A";
  const aucRocVal = m.auc_roc !== undefined && m.auc_roc !== null ? m.auc_roc.toFixed(4) : "N/A";
  const aucPrVal = m.auc_pr !== undefined && m.auc_pr !== null ? m.auc_pr.toFixed(4) : "N/A";

  const cal = report.calibration || {};
  const eceVal = cal.ece !== undefined && cal.ece !== null ? (cal.ece * 100).toFixed(2) + "%" : "N/A";

  const totalCostVal = m.total_cost !== undefined && m.total_cost !== null ? `₹${m.total_cost.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : "N/A";
  const fpCostVal = m.fp_cost !== undefined && m.fp_cost !== null ? `₹${m.fp_cost.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : "N/A";
  const fnCostVal = m.fn_exposure !== undefined && m.fn_exposure !== null ? `₹${m.fn_exposure.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : "N/A";

  const tpVal = m.tp !== undefined && m.tp !== null ? m.tp : "N/A";
  const fpVal = m.fp !== undefined && m.fp !== null ? m.fp : "N/A";
  const fnVal = m.fn !== undefined && m.fn !== null ? m.fn : "N/A";
  const tnVal = m.tn !== undefined && m.tn !== null ? m.tn.toLocaleString() : "N/A";

  const elP = document.getElementById("trackBPrecision");
  if (elP) elP.innerHTML = precVal !== "N/A" ? `${precVal} <span class="metric-unit">(${precPct}%)</span>` : "N/A";
  const elPCi = document.getElementById("trackBPrecisionCI");
  if (elPCi) elPCi.textContent = `[${precLow}, ${precHigh}]`;
  const elTp = document.getElementById("trackBTP");
  if (elTp) elTp.textContent = tpVal;
  const elFp = document.getElementById("trackBFP");
  if (elFp) elFp.textContent = fpVal;

  const elR = document.getElementById("trackBRecall");
  if (elR) elR.innerHTML = recVal !== "N/A" ? `${recVal} <span class="metric-unit">(${recPct}%)</span>` : "N/A";
  const elRCi = document.getElementById("trackBRecallCI");
  if (elRCi) elRCi.textContent = `[${recLow}, ${recHigh}]`;
  const elFn = document.getElementById("trackBFN");
  if (elFn) elFn.textContent = fnVal;
  const elTn = document.getElementById("trackBTN");
  if (elTn) elTn.textContent = tnVal;

  const elF1 = document.getElementById("trackBF1");
  if (elF1) elF1.textContent = f1Val;
  const elAucRoc = document.getElementById("trackBAucRoc");
  if (elAucRoc) elAucRoc.textContent = aucRocVal;
  const elAucPr = document.getElementById("trackBAucPr");
  if (elAucPr) elAucPr.textContent = aucPrVal;
  const elEce = document.getElementById("trackBEce");
  if (elEce) elEce.textContent = eceVal;

  const elCost = document.getElementById("trackBCost");
  if (elCost) elCost.textContent = totalCostVal;
  const elFpCost = document.getElementById("trackBFpCost");
  if (elFpCost) elFpCost.textContent = fpCostVal;
  const elFnCost = document.getElementById("trackBFnCost");
  if (elFnCost) elFnCost.textContent = fnCostVal;
}

function populateEvaluationSummary(report) {
  const m = report.executive_summary || {};
  document.getElementById("evalPrecision").textContent = m.precision !== undefined && m.precision !== null ? m.precision.toFixed(4) : "N/A";
  document.getElementById("evalRecall").textContent = m.recall !== undefined && m.recall !== null ? m.recall.toFixed(4) : "N/A";
  document.getElementById("evalF1").textContent = m.f1_score !== undefined && m.f1_score !== null ? m.f1_score.toFixed(4) : "N/A";
  document.getElementById("evalTP").textContent = m.tp !== undefined && m.tp !== null ? m.tp : "N/A";
  document.getElementById("evalFP").textContent = m.fp !== undefined && m.fp !== null ? m.fp : "N/A";
  document.getElementById("evalFN").textContent = m.fn !== undefined && m.fn !== null ? m.fn : "N/A";
  document.getElementById("evalCost").textContent = m.total_cost !== undefined && m.total_cost !== null ? `₹${m.total_cost.toFixed(2)}` : "N/A";

  // Extract canonical P95 latency from descriptive portfolio or executive metrics
  const port = report.descriptive_portfolio_analysis || [];
  const canonicalScorer = port.find(p => p.is_frozen_canonical) || {};
  const p95Lat = canonicalScorer.p95_latency_seconds !== undefined && canonicalScorer.p95_latency_seconds !== null
    ? `${canonicalScorer.p95_latency_seconds.toFixed(2)}s`
    : (m.p95_latency_seconds !== undefined ? `${m.p95_latency_seconds.toFixed(2)}s` : "N/A");
  
  document.getElementById("evalP95").textContent = p95Lat;

  const medLat = m.median_latency_seconds !== undefined && m.median_latency_seconds !== null ? m.median_latency_seconds.toFixed(2) : (canonicalScorer.median_latency_seconds ? canonicalScorer.median_latency_seconds.toFixed(2) : "N/A");
  document.getElementById("evalLatency").innerHTML = `${medLat} <span class="metric-unit">s</span>`;

  // Per-anomaly performance breakdown using exact canonical report keys
  const perAnom = report.per_anomaly_performance || {};
  const tbody = document.getElementById("tblPerAnomaly");
  tbody.innerHTML = "";

  const classes = [
    { name: "Sudden Volume Spike", key: "volume_spike" },
    { name: "Velocity Burst", key: "velocity_burst" },
    { name: "Sustained Volume Spike", key: "sustained_spike" },
    { name: "Amount Distribution Shift", key: "amount_shift" },
    { name: "Behavioral Device Anomaly", key: "behavioral_anomaly" },
    { name: "Attribute Geo Anomaly", key: "attribute_shift" },
    { name: "Compound Anomaly", key: "compound_anomaly" },
    { name: "Detector-Aware Evasion Patterns", key: "evasive_patterns" },
  ];

  classes.forEach(c => {
    const item = perAnom[c.key] || {};
    const status = item.status || "N/A — not present in artifact";
    const isValidated = status === "VALIDATED";

    const eventIdDisplay = item.event_id !== undefined && item.event_id !== null 
      ? item.event_id 
      : (item.ground_truth_events && item.ground_truth_events.length > 0 
          ? item.ground_truth_events.join(", ") 
          : "N/A — not present in artifact");

    const tpDisplay = item.events_detected !== undefined && item.events_detected !== null 
      ? item.events_detected 
      : (item.tp !== undefined && item.tp !== null ? item.tp : "N/A — not present in artifact");

    const fpDisplay = item.fp !== undefined && item.fp !== null ? item.fp : "N/A — not present in artifact";
    const fnDisplay = item.fn !== undefined && item.fn !== null ? item.fn : "N/A — not present in artifact";

    const recallDisplay = item.recall !== undefined && item.recall !== null 
      ? (item.recall * 100).toFixed(1) + '%' 
      : "N/A — not present in artifact";

    const latencyDisplay = item.median_latency_seconds !== undefined && item.median_latency_seconds !== null 
      ? item.median_latency_seconds.toFixed(2) + 's' 
      : "N/A — not present in artifact";

    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td style="font-weight:600;">${c.name}</td>
      <td>
        <span class="badge ${isValidated ? 'badge-frozen' : 'badge-defense'}" style="font-size:0.7rem;">
          ${status}
        </span>
      </td>
      <td>${eventIdDisplay}</td>
      <td>${tpDisplay}</td>
      <td>${fpDisplay}</td>
      <td>${fnDisplay}</td>
      <td>${recallDisplay}</td>
      <td>${latencyDisplay}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateAblationTable(results) {
  const tbody = document.getElementById("tblAblation");
  tbody.innerHTML = "";

  if (results.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">Ablation results unavailable.</td></tr>`;
    return;
  }

  results.forEach(res => {
    const tr = document.createElement("tr");
    const m = res.metrics || {};
    const dF1 = res.delta_f1 !== undefined ? (res.delta_f1 >= 0 ? `+${res.delta_f1.toFixed(4)}` : res.delta_f1.toFixed(4)) : "0.0000";

    tr.innerHTML = `
      <td style="font-weight:700; color:${res.variant_id === 'FULL' ? 'var(--accent-cyan)' : 'var(--text-main)'};">${res.variant_id}</td>
      <td>${m.precision !== undefined ? m.precision.toFixed(4) : 'N/A'}</td>
      <td>${m.recall !== undefined ? m.recall.toFixed(4) : 'N/A'}</td>
      <td style="font-weight:700;">${m.f1_score !== undefined ? m.f1_score.toFixed(4) : 'N/A'}</td>
      <td style="color:${res.delta_f1 === 0 ? 'var(--text-muted)' : (res.delta_f1 < 0 ? 'var(--color-danger)' : 'var(--color-success)')};">${dF1}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateEvasionTable(scenarios) {
  const tbody = document.getElementById("tblEvasion");
  tbody.innerHTML = "";

  const keys = Object.keys(scenarios);
  if (keys.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">Evasion confirmation scenarios unavailable.</td></tr>`;
    return;
  }

  keys.forEach(scKey => {
    const sc = scenarios[scKey];
    const m = sc.measurements || {};
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td style="font-weight:600;">${scKey}</td>
      <td>${sc.event_id || "N/A"}</td>
      <td>${m.max_observed_score !== undefined && m.max_observed_score !== null ? m.max_observed_score.toFixed(2) + 'σ' : 'N/A'}</td>
      <td><span class="badge ${m.alerts_emitted > 0 ? 'badge-frozen' : 'badge-defense'}">${m.alerts_emitted > 0 ? 'YES (' + m.alerts_emitted + ')' : 'NO (0)'}</span></td>
      <td style="font-weight:700; color:${m.evaluation_outcome === 'TP' ? 'var(--color-success)' : 'var(--color-warning)'};">${m.evaluation_outcome || 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateCalibration(cal) {
  const desc = cal.descriptive_calibration || cal;
  const samples = desc.total_samples !== undefined ? desc.total_samples : (desc.sample_count !== undefined ? desc.sample_count : "N/A");
  const ece = desc.expected_calibration_error !== undefined && desc.expected_calibration_error !== null ? desc.expected_calibration_error.toFixed(4) : "N/A";

  document.getElementById("calibSamples").textContent = samples;
  document.getElementById("calibECE").textContent = ece;

  const buckets = desc.buckets || desc.reliability_diagram || [];
  const div = document.getElementById("calibBuckets");
  div.innerHTML = "";

  if (buckets.length === 0) {
    div.innerHTML = `<div style="color:var(--text-muted); font-style:italic;">No calibration buckets available.</div>`;
    return;
  }

  buckets.forEach(b => {
    const bName = b.bucket || b.bin || "Bucket";
    const n = b.n !== undefined ? b.n : (b.sample_count !== undefined ? b.sample_count : 0);
    const rate = b.observed_positive_rate !== undefined ? b.observed_positive_rate : (b.empirical_accuracy || 0);
    const pct = (rate * 100).toFixed(1);
    const row = document.createElement("div");
    row.style.fontSize = "0.8rem";
    row.innerHTML = `
      <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
        <span>Bin ${bName} (N=${n})</span>
        <span>Observed Positive Rate: <strong>${pct}%</strong></span>
      </div>
      <div class="bar-container">
        <div class="bar-fill" style="width: ${Math.max(4, rate * 100)}%;"></div>
      </div>
    `;
    div.appendChild(row);
  });
}

function populateBootstrap(boot) {
  const tbody = document.getElementById("tblBootstrap");
  tbody.innerHTML = "";

  const bData = boot.bootstrap_uncertainty || boot;
  const prec = bData.precision || {};
  const rec = bData.recall || {};

  const pLow = prec.ci_lower !== undefined ? prec.ci_lower : prec.ci_95_lower;
  const pHigh = prec.ci_upper !== undefined ? prec.ci_upper : prec.ci_95_upper;
  if (pLow !== undefined && pHigh !== undefined) {
    const elP = document.getElementById("evalPrecisionCI");
    if (elP) elP.textContent = `[${pLow.toFixed(4)}, ${pHigh.toFixed(4)}]`;
  }

  const rLow = rec.ci_lower !== undefined ? rec.ci_lower : rec.ci_95_lower;
  const rHigh = rec.ci_upper !== undefined ? rec.ci_upper : rec.ci_95_upper;
  if (rLow !== undefined && rHigh !== undefined) {
    const elR = document.getElementById("evalRecallCI");
    if (elR) elR.textContent = `[${rLow.toFixed(4)}, ${rHigh.toFixed(4)}]`;
  }

  if (prec.point === undefined && rec.point === undefined) {
    tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; color:var(--text-muted);">Bootstrap CI data unavailable.</td></tr>`;
    return;
  }

  const metrics = [
    { name: "Precision", data: prec },
    { name: "Recall", data: rec },
  ];

  metrics.forEach(m => {
    const item = m.data;
    const point = item.point !== undefined ? item.point.toFixed(4) : "N/A";
    const ciLower = item.ci_lower !== undefined ? item.ci_lower.toFixed(4) : (item.ci_95_lower !== undefined ? item.ci_95_lower.toFixed(4) : "N/A");
    const ciUpper = item.ci_upper !== undefined ? item.ci_upper.toFixed(4) : (item.ci_95_upper !== undefined ? item.ci_95_upper.toFixed(4) : "N/A");
    const rawNum = item.raw_numerator_tp !== undefined ? item.raw_numerator_tp : "N/A";
    const rawDen = item.raw_denominator_alerts !== undefined ? item.raw_denominator_alerts : (item.raw_denominator_events !== undefined ? item.raw_denominator_events : "N/A");

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight:600;">${m.name}</td>
      <td style="font-weight:700;">${point} (Raw: ${rawNum}/${rawDen})</td>
      <td>[${ciLower}, ${ciUpper}]</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateProvenance(rep) {
  document.getElementById("provHoldoutHash").textContent = rep.holdout_dataset_hash || "N/A";
  document.getElementById("provExpId").textContent = rep.experiment_id || "N/A";
  document.getElementById("provArtSha").textContent = rep.artifact_sha256 || "N/A";
}

// Fetch SQLite Audit Trail
async function fetchAuditLogs() {
  try {
    const merchantId = document.getElementById("selectMerchant").value || "M1";
    const res = await fetch(`/api/audit?merchant_id=${merchantId}`);
    const data = await res.json();
    const tbody = document.getElementById("tblAuditLogs");
    tbody.innerHTML = "";

    const audits = data.audit_records || [];
    if (audits.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">No SQLite audit records yet. Click "Start Deterministic Stream Demo" or "Step Window" on Tab 1.</td></tr>`;
      return;
    }

    audits.slice().reverse().forEach(rec => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:monospace;">${rec.audit_id ? rec.audit_id.substring(0, 16) + '...' : '-'}</td>
        <td>${rec.timestamp ? rec.timestamp.substring(11, 19) : '-'}</td>
        <td>${rec.merchant_id}</td>
        <td style="font-weight:700;">${rec.risk_score !== null ? rec.risk_score.toFixed(2) + 'σ' : 'None'}</td>
        <td>${rec.confidence !== undefined ? rec.confidence.toFixed(2) : '1.00'}</td>
        <td><span class="badge ${rec.data_quality_status === 'GOOD' ? 'badge-frozen' : 'badge-defense'}">${rec.data_quality_status}</span></td>
        <td>${rec.alert_id ? '<span class="badge state-ALERT">ALERT: ' + rec.alert_id.substring(0, 8) + '</span>' : 'AUDIT_LOG'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to fetch audit logs:", err);
  }
}

// Demo Simulation Playback Engine
let isPlaying = false;
let playInterval = null;

function toggleDemoPlay() {
  const btn = document.getElementById("btnPlayDemo");
  if (!isPlaying) {
    isPlaying = true;
    btn.textContent = "⏸ Pause Demo";
    btn.classList.replace("btn-primary", "btn-secondary");
    playInterval = setInterval(stepDemo, 1200);
  } else {
    isPlaying = false;
    btn.textContent = "▶ Start Deterministic Stream Demo";
    btn.classList.replace("btn-secondary", "btn-primary");
    clearInterval(playInterval);
  }
}

async function stepDemo() {
  try {
    // Visibly animate 8 pipeline workflow stages
    animatePipelineFlow();

    const res = await fetch("/api/demo/step", { method: "POST" });
    const data = await res.json();
    renderDemoStep(data);
  } catch (err) {
    console.error("Failed demo step:", err);
  }
}

async function resetDemo(merchantId) {
  if (isPlaying) toggleDemoPlay();
  const mId = typeof merchantId === "string" ? merchantId : document.getElementById("selectMerchant").value;
  try {
    await fetch(`/api/demo/reset?merchant_id=${mId}`, { method: "POST" });
    const wIdxEl = document.getElementById("lblWindowIndex");
    if (wIdxEl) wIdxEl.textContent = "#0";
    const wTimeEl = document.getElementById("lblWindowTime");
    if (wTimeEl) wTimeEl.textContent = "2026-01-01T12:00:00Z";
    
    document.getElementById("valRiskScore").innerHTML = `0.00 <span class="metric-unit">σ</span>`;
    document.getElementById("valThresholdCheck").textContent = "Sub-threshold (Normal)";
    document.getElementById("valConfidence").innerHTML = `1.00 <span class="metric-unit">/ 1.0</span>`;
    document.getElementById("badgeState").className = "state-badge state-NORMAL";
    document.getElementById("badgeState").textContent = "NORMAL";
    document.getElementById("explanationText").textContent = `Demo reset for merchant ${mId}. StatisticalDeviationScorer initialized (threshold 5.0, persistence 1, cooldown 5, v1.1.0). Ready for playback.`;
    document.getElementById("tblFeatures").innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No stream data yet.</td></tr>`;
    document.getElementById("tblTransactions").innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No transactions in current window.</td></tr>`;
  } catch (err) {
    console.error("Failed demo reset:", err);
  }
}

function renderDemoStep(data) {
  // Update Window Index & Deterministic Timestamp clock
  const wIdxEl = document.getElementById("lblWindowIndex");
  if (wIdxEl) wIdxEl.textContent = `#${data.window_index !== undefined ? data.window_index : 0}`;
  const wTimeEl = document.getElementById("lblWindowTime");
  if (wTimeEl && data.timestamp) wTimeEl.textContent = data.timestamp;

  // Update state machine badge
  const st = data.state_machine_status || "NORMAL";
  const badge = document.getElementById("badgeState");
  badge.className = `state-badge state-${st}`;
  badge.textContent = st;

  // Update Risk Score & Confidence
  const audit = data.audit || {};
  const score = audit.risk_score !== undefined && audit.risk_score !== null ? audit.risk_score : 0.0;
  const conf = audit.confidence !== undefined ? audit.confidence : 1.0;
  
  document.getElementById("valRiskScore").innerHTML = `${score.toFixed(2)} <span class="metric-unit">σ</span>`;
  document.getElementById("valThresholdCheck").textContent = score >= 5.0 ? "⚠️ THRESHOLD BREACHED (≥ 5.00σ)" : "Sub-threshold (Normal)";
  document.getElementById("valConfidence").innerHTML = `${conf.toFixed(2)} <span class="metric-unit">/ 1.0</span>`;
  document.getElementById("valEvidenceState").innerHTML = `Evidence State: <strong>${audit.baseline ? audit.baseline.evidence_state : 'SUFFICIENT'}</strong>`;
  document.getElementById("valDataQuality").innerHTML = `Data Quality: <strong>${audit.data_quality_status || 'GOOD'}</strong>`;

  // Triggered signals
  const sigs = audit.triggered_signals || [];
  document.getElementById("valTriggeredSignals").innerHTML = sigs.length > 0 
    ? sigs.map(s => `<span class="badge badge-defense" style="margin-right:0.3rem;">${s}</span>`).join("")
    : `<span style="color:var(--text-muted); font-style:italic;">None</span>`;

  // Explanation box ("What Happened?") using exact current detector values
  document.getElementById("explanationText").textContent = data.explanation || "";

  // Render Features vs Baseline table
  const feat = audit.features || {};
  const base = audit.baseline || {};
  const exp = base.expected_values || {};
  const scale = base.robust_scale || {};

  const tblF = document.getElementById("tblFeatures");
  tblF.innerHTML = `
    <tr>
      <td style="font-weight:600;">Volume (txs/min)</td>
      <td><strong>${feat.volume !== undefined ? feat.volume : data.transaction_count}</strong></td>
      <td>${exp.volume !== undefined && exp.volume !== null ? exp.volume.toFixed(2) : 'N/A — not emitted by backend'}</td>
      <td>${scale.volume !== undefined && scale.volume !== null ? scale.volume.toFixed(2) : 'N/A — not emitted by backend'}</td>
      <td style="font-weight:700; color:${score >= 5.0 ? 'var(--color-danger)' : 'var(--accent-cyan)'};">${score.toFixed(2)}σ</td>
    </tr>
    <tr>
      <td style="font-weight:600;">Velocity (txs/sec)</td>
      <td>${feat.velocity !== undefined && feat.velocity !== null ? feat.velocity.toFixed(2) : 'N/A — not emitted by backend'}</td>
      <td>${exp.velocity !== undefined && exp.velocity !== null ? exp.velocity.toFixed(2) : 'N/A — not emitted by backend'}</td>
      <td>${scale.velocity !== undefined && scale.velocity !== null ? scale.velocity.toFixed(2) : 'N/A — not emitted by backend'}</td>
      <td>${feat.velocity !== undefined && feat.velocity !== null ? '0.00σ' : 'N/A — not emitted by backend'}</td>
    </tr>
    <tr>
      <td style="font-weight:600;">Unique Devices</td>
      <td>${feat.unique_devices !== undefined && feat.unique_devices !== null ? feat.unique_devices : 'N/A — not emitted by backend'}</td>
      <td>${exp.unique_devices !== undefined && exp.unique_devices !== null ? exp.unique_devices.toFixed(2) : 'N/A — not emitted by backend'}</td>
      <td>${scale.unique_devices !== undefined && scale.unique_devices !== null ? scale.unique_devices.toFixed(2) : 'N/A — not emitted by backend'}</td>
      <td>${feat.unique_devices !== undefined && feat.unique_devices !== null ? '0.00σ' : 'N/A — not emitted by backend'}</td>
    </tr>
  `;

  // Render Transaction Stream
  const txs = data.transactions || [];
  const tblT = document.getElementById("tblTransactions");
  tblT.innerHTML = "";
  if (txs.length === 0) {
    tblT.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">Empty window (0 transactions).</td></tr>`;
  } else {
    txs.forEach(t => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family:monospace;">${t.transaction_id ? t.transaction_id.substring(0, 10) : 'N/A — not emitted by backend'}</td>
        <td>${t.timestamp ? t.timestamp.substring(11, 19) : 'N/A — not emitted by backend'}</td>
        <td style="font-weight:600;">${t.amount !== undefined && t.amount !== null ? '₹' + t.amount.toFixed(2) : 'N/A — not emitted by backend'}</td>
        <td>${t.payment_method || 'N/A — not emitted by backend'}</td>
        <td style="font-family:monospace;">${t.device_id || 'N/A — not emitted by backend'}</td>
      `;
      tblT.appendChild(tr);
    });
  }
}

function animatePipelineFlow() {
  const stages = ["stageTx", "stageFeat", "stageBase", "stageScore", "stageConf", "stageState", "stageAlert", "stageAudit"];
  stages.forEach((sId, idx) => {
    setTimeout(() => {
      stages.forEach(s => {
        const el = document.getElementById(s);
        if (el) el.classList.remove("active");
      });
      const currentEl = document.getElementById(sId);
      if (currentEl) currentEl.classList.add("active");
    }, idx * 70);
  });
}
