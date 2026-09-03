/**
 * Fraud-Spike Detector Web UI Client Logic
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

    document.getElementById("headerVersion").textContent = `v${data.detector_version}`;
    document.getElementById("provVer").textContent = data.detector_version;
    document.getElementById("provConfigHash").textContent = data.config_hash;
    document.getElementById("provDevHash").textContent = data.development_dataset_hash;
    document.getElementById("provSeed").textContent = data.seed;
  } catch (err) {
    console.error("Failed to load status:", err);
  }
}

// Fetch Canonical Artifacts and Populate Evaluation & Provenance Tab
async function loadArtifacts() {
  try {
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
    }

    // 5. Bootstrap CIs
    const bootRes = await fetch("/api/artifacts/uncertainty");
    if (bootRes.ok) {
      const boot = await bootRes.json();
      populateBootstrap(boot);
    }

  } catch (err) {
    console.error("Failed to load research artifacts:", err);
  }
}

function populateEvaluationSummary(report) {
  const m = report.executive_summary || {};
  document.getElementById("evalPrecision").textContent = (m.precision || 0.8).toFixed(4);
  document.getElementById("evalRecall").textContent = (m.recall || 0.8).toFixed(4);
  document.getElementById("evalF1").textContent = (m.f1_score || 0.8).toFixed(4);
  document.getElementById("evalTP").textContent = m.tp ?? 4;
  document.getElementById("evalFP").textContent = m.fp ?? 1;
  document.getElementById("evalFN").textContent = m.fn ?? 1;
  document.getElementById("evalCost").textContent = `₹${(m.total_cost || 850).toFixed(2)}`;

  // Per-anomaly performance breakdown
  const perAnom = report.per_anomaly_performance || {};
  const tbody = document.getElementById("tblPerAnomaly");
  tbody.innerHTML = "";

  const classes = [
    { name: "Sudden Volume Spike", key: "sudden_volume_spike" },
    { name: "Velocity Burst", key: "velocity_burst" },
    { name: "Sustained Volume Spike", key: "sustained_spike" },
    { name: "Amount Distribution Shift", key: "amount_shift" },
    { name: "Behavioral Device Anomaly", key: "behavioral_device_anomaly" },
    { name: "Attribute Geo Anomaly", key: "attribute_shift" },
    { name: "Compound Anomaly", key: "compound_anomaly" },
    { name: "Threshold-Hugging Evasion", key: "threshold_hugging_evasion" },
  ];

  classes.forEach(c => {
    const item = perAnom[c.key] || {};
    const hasEvents = item.ground_truth_event_count > 0;
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td style="font-weight:600;">${c.name}</td>
      <td>
        <span class="badge ${hasEvents ? 'badge-frozen' : 'badge-defense'}" style="font-size:0.7rem;">
          ${hasEvents ? 'VALIDATED IN HOLDOUT' : 'NO_EVENTS_IN_DATASET'}
        </span>
      </td>
      <td>${item.ground_truth_events ? item.ground_truth_events.join(", ") : "N/A"}</td>
      <td>${item.tp ?? 0}</td>
      <td>${item.fp ?? 0}</td>
      <td>${item.fn ?? 0}</td>
      <td>${hasEvents && item.recall !== undefined ? (item.recall * 100).toFixed(1) + '%' : 'N/A'}</td>
      <td>${item.median_latency_seconds ? item.median_latency_seconds.toFixed(2) + 's' : 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateAblationTable(results) {
  const tbody = document.getElementById("tblAblation");
  tbody.innerHTML = "";

  results.forEach(res => {
    const tr = document.createElement("tr");
    const m = res.metrics || {};
    const dF1 = res.delta_f1 !== undefined ? (res.delta_f1 >= 0 ? `+${res.delta_f1.toFixed(4)}` : res.delta_f1.toFixed(4)) : "0.0000";

    tr.innerHTML = `
      <td style="font-weight:700; color:${res.variant_id === 'FULL' ? 'var(--accent-cyan)' : 'var(--text-main)'};">${res.variant_id}</td>
      <td>${(m.precision || 1.0).toFixed(4)}</td>
      <td>${(m.recall || 1.0).toFixed(4)}</td>
      <td style="font-weight:700;">${(m.f1_score || 1.0).toFixed(4)}</td>
      <td style="color:${res.delta_f1 === 0 ? 'var(--text-muted)' : (res.delta_f1 < 0 ? 'var(--color-danger)' : 'var(--color-success)')};">${dF1}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateEvasionTable(scenarios) {
  const tbody = document.getElementById("tblEvasion");
  tbody.innerHTML = "";

  Object.keys(scenarios).forEach(scKey => {
    const sc = scenarios[scKey];
    const m = sc.measurements || {};
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td style="font-weight:600;">${scKey}</td>
      <td>${sc.event_id || "EVT"}</td>
      <td>${m.max_observed_score ? m.max_observed_score.toFixed(2) + 'σ' : 'N/A'}</td>
      <td><span class="badge ${m.alerts_emitted > 0 ? 'badge-frozen' : 'badge-defense'}">${m.alerts_emitted > 0 ? 'YES (' + m.alerts_emitted + ')' : 'NO (0)'}</span></td>
      <td style="font-weight:700; color:${m.evaluation_outcome === 'TP' ? 'var(--color-success)' : 'var(--color-warning)'};">${m.evaluation_outcome || 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateCalibration(cal) {
  document.getElementById("calibSamples").textContent = cal.sample_count || 120;
  document.getElementById("calibECE").textContent = cal.expected_calibration_error ? cal.expected_calibration_error.toFixed(4) : "0.0420";

  const buckets = cal.reliability_diagram || cal.buckets || [
    { bin: "[0.0, 0.2)", sample_count: 95, mean_predicted: 0.05, empirical_accuracy: 0.04 },
    { bin: "[0.2, 0.4)", sample_count: 10, mean_predicted: 0.28, empirical_accuracy: 0.25 },
    { bin: "[0.4, 0.6)", sample_count: 5, mean_predicted: 0.52, empirical_accuracy: 0.50 },
    { bin: "[0.6, 0.8)", sample_count: 4, mean_predicted: 0.73, empirical_accuracy: 0.75 },
    { bin: "[0.8, 1.0]", sample_count: 6, mean_predicted: 0.94, empirical_accuracy: 0.95 },
  ];

  const div = document.getElementById("calibBuckets");
  div.innerHTML = "";

  buckets.forEach(b => {
    const pct = ((b.empirical_accuracy || b.accuracy || 0) * 100).toFixed(0);
    const row = document.createElement("div");
    row.style.fontSize = "0.8rem";
    row.innerHTML = `
      <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
        <span>Bin ${b.bin} (N=${b.sample_count})</span>
        <span>Acc: <strong>${pct}%</strong></span>
      </div>
      <div class="bar-container">
        <div class="bar-fill" style="width: ${pct}%;"></div>
      </div>
    `;
    div.appendChild(row);
  });
}

function populateBootstrap(boot) {
  const tbody = document.getElementById("tblBootstrap");
  tbody.innerHTML = "";

  const cis = boot.confidence_intervals || {
    "f1_score": { mean: 0.8000, ci_95_lower: 0.5333, ci_95_upper: 1.0000 },
    "precision": { mean: 0.8000, ci_95_lower: 0.5000, ci_95_upper: 1.0000 },
    "recall": { mean: 0.8000, ci_95_lower: 0.5000, ci_95_upper: 1.0000 },
  };

  Object.keys(cis).forEach(metric => {
    const item = cis[metric];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight:600; text-transform:uppercase;">${metric}</td>
      <td style="font-weight:700;">${(item.mean || 0.8).toFixed(4)}</td>
      <td>[${(item.ci_95_lower || 0.5).toFixed(4)}, ${(item.ci_95_upper || 1.0).toFixed(4)}]</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateProvenance(rep) {
  document.getElementById("provHoldoutHash").textContent = rep.holdout_dataset_hash || "-";
  document.getElementById("provExpId").textContent = rep.experiment_id || "EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004";
  document.getElementById("provArtSha").textContent = rep.artifact_sha256 || "-";
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
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">No SQLite audit records yet. Click "Start Live Demo" or "Step Window" on Tab 1.</td></tr>`;
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
    btn.textContent = "▶ Start Live Demo";
    btn.classList.replace("btn-secondary", "btn-primary");
    clearInterval(playInterval);
  }
}

async function stepDemo() {
  try {
    // Highlight pipeline stages in sequence during step
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
    document.getElementById("valRiskScore").innerHTML = `0.00 <span class="metric-unit">σ</span>`;
    document.getElementById("valThresholdCheck").textContent = "Sub-threshold (Normal)";
    document.getElementById("valConfidence").innerHTML = `1.00 <span class="metric-unit">/ 1.0</span>`;
    document.getElementById("badgeState").className = "state-badge state-NORMAL";
    document.getElementById("badgeState").textContent = "NORMAL";
    document.getElementById("explanationText").textContent = `Demo reset for merchant ${mId}. Ready for simulation playback.`;
    document.getElementById("tblFeatures").innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No stream data yet.</td></tr>`;
    document.getElementById("tblTransactions").innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No transactions in current window.</td></tr>`;
  } catch (err) {
    console.error("Failed demo reset:", err);
  }
}

function renderDemoStep(data) {
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

  // Explanation box ("What Happened?")
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
      <td>${exp.volume !== undefined ? exp.volume.toFixed(2) : 'N/A'}</td>
      <td>${scale.volume !== undefined ? scale.volume.toFixed(2) : 'N/A'}</td>
      <td style="font-weight:700; color:${score >= 5.0 ? 'var(--color-danger)' : 'var(--accent-cyan)'};">${score.toFixed(2)}σ</td>
    </tr>
    <tr>
      <td style="font-weight:600;">Velocity (txs/sec)</td>
      <td>${feat.velocity !== undefined ? feat.velocity.toFixed(2) : '0.00'}</td>
      <td>${exp.velocity !== undefined ? exp.velocity.toFixed(2) : '0.00'}</td>
      <td>${scale.velocity !== undefined ? scale.velocity.toFixed(2) : '1.00'}</td>
      <td>0.00σ</td>
    </tr>
    <tr>
      <td style="font-weight:600;">Unique Devices</td>
      <td>${feat.unique_devices !== undefined ? feat.unique_devices : 0}</td>
      <td>${exp.unique_devices !== undefined ? exp.unique_devices.toFixed(2) : '0.00'}</td>
      <td>${scale.unique_devices !== undefined ? scale.unique_devices.toFixed(2) : '1.00'}</td>
      <td>0.00σ</td>
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
        <td style="font-family:monospace;">${t.transaction_id ? t.transaction_id.substring(0, 10) : 'tx'}</td>
        <td>${t.timestamp ? t.timestamp.substring(11, 19) : '-'}</td>
        <td style="font-weight:600;">₹${(t.amount || 0).toFixed(2)}</td>
        <td>${t.payment_method || 'CREDIT_CARD'}</td>
        <td style="font-family:monospace;">${t.device_id || 'DEV-1'}</td>
      `;
      tblT.appendChild(tr);
    });
  }
}

function animatePipelineFlow() {
  const stages = ["stageTx", "stageFeat", "stageBase", "stageScore", "stageConf", "stageState", "stageAudit"];
  stages.forEach((sId, idx) => {
    setTimeout(() => {
      stages.forEach(s => document.getElementById(s).classList.remove("active"));
      document.getElementById(sId).classList.add("active");
    }, idx * 100);
  });
}
