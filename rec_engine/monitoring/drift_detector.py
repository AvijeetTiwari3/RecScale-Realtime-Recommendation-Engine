"""
rec_engine/monitoring/drift_detector.py
=======================================
Online Data & Concept Drift Monitoring (Evidently AI / Statistical PSI & KS-Test).
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline and production streaming distributions.
    Interpretation:
        PSI < 0.1  : No significant change / stable.
        0.1 <= PSI < 0.2 : Moderate drift.
        PSI >= 0.2 : Significant drift -> triggers model/feature retraining.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * num_buckets)
    actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * num_buckets)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


class DriftDetector:
    """
    Real-time statistical drift monitor tracking streaming feature drift and CTR prediction drift.
    """

    def __init__(self, baseline_predictions: Optional[List[float]] = None):
        self.baseline_predictions = np.array(baseline_predictions or np.random.uniform(0.1, 0.9, 500))
        self.streaming_buffer: List[float] = []
        self.buffer_limit = 200

    def record_prediction(self, ctr_score: float):
        self.streaming_buffer.append(ctr_score)
        if len(self.streaming_buffer) > self.buffer_limit:
            self.streaming_buffer.pop(0)

    def evaluate_drift(self) -> Dict[str, Any]:
        """
        Runs PSI and Kolmogorov-Smirnov test over the streaming buffer.
        """
        if len(self.streaming_buffer) < 20:
            return {"status": "insufficient_data", "psi": 0.0, "ks_p_value": 1.0, "drift_detected": False}

        current_data = np.array(self.streaming_buffer)
        psi_score = calculate_psi(self.baseline_predictions, current_data)
        
        ks_stat, p_val = stats.ks_2samp(self.baseline_predictions, current_data)
        drift_detected = psi_score >= 0.2 or p_val < 0.05

        return {
            "psi": round(psi_score, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(p_val), 4),
            "drift_detected": bool(drift_detected),
            "severity": "CRITICAL" if psi_score >= 0.2 else ("MODERATE" if psi_score >= 0.1 else "HEALTHY")
        }
