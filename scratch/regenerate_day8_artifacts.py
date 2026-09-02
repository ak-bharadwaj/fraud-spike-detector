from pathlib import Path
from src.evaluation.freeze import load_freeze_record
from src.evaluation.holdout import load_locked_holdout_data
from src.evaluation.holdout_execution import (
    execute_single_pass_holdout,
    compute_per_anomaly_holdout_metrics,
    compute_descriptive_holdout_calibration,
    compute_bootstrap_uncertainty,
    execute_portfolio_comparison,
    build_canonical_holdout_evasion_results,
    build_canonical_holdout_drift_results,
    save_day8_research_artifacts,
)

from src.evaluation.sweeps import load_development_data, run_ewma_precision_latency_tradeoff_sweep

freeze_record = load_freeze_record("config/freeze_record.json")
manifest, txs, gts = load_locked_holdout_data("data/holdout")

dev_txs, dev_gts = load_development_data("data/development")
ewma_sweep = run_ewma_precision_latency_tradeoff_sweep(dev_txs, dev_gts)

m, alerts, scores = execute_single_pass_holdout(txs, gts, freeze_record, explicit_evaluation_mode=True)
per_ano = compute_per_anomaly_holdout_metrics(alerts, gts)
calib = compute_descriptive_holdout_calibration(scores, gts)
boot = compute_bootstrap_uncertainty(alerts, gts, n_resamples=1000, seed=42)
port = execute_portfolio_comparison(txs, gts, freeze_record)

evasion_res = build_canonical_holdout_evasion_results(freeze_record, manifest)
drift_res = build_canonical_holdout_drift_results(freeze_record, manifest)

save_day8_research_artifacts(
    base_artifact_dir="artifacts",
    freeze_record=freeze_record,
    holdout_manifest=manifest,
    holdout_metrics=m,
    per_anomaly_metrics=per_ano,
    calibration_results=calib,
    bootstrap_results=boot,
    portfolio_results=port,
    evasion_results=evasion_res,
    drift_results=drift_res,
    ewma_tradeoff_results=ewma_sweep,
    experiment_id="EXP-DAY8-HOLDOUT-RECONSTRUCTED-003",
    execution_commit="bc29c36",
    artifact_finalization_commit="5841ddb",
    prior_artifact_commit="049caf5",
    historical_artifact_chain=["20bf655", "775e779", "cc2872b", "e28d6d3", "f21ddeb", "26837b7", "bc29c36", "049caf5", "5841ddb"],
)

from src.generator.degradation import execute_data_quality_characterization
execute_data_quality_characterization(base_artifact_dir="artifacts", seed=42)

print("Artifacts successfully regenerated!")
