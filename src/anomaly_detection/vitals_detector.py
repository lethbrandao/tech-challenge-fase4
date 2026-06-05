"""
Módulo 3 — Detecção de Anomalias em Sinais Vitais
Aplica Isolation Forest sobre séries temporais de batimentos,
pressão arterial, oxigenação e evolução de prescrições.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 0.15))
REPORTS_PATH = Path(os.getenv("REPORTS_PATH", "reports"))

# Faixas de referência clínica (para enriquecer o relatório)
CLINICAL_RANGES = {
    "heart_rate":       {"min": 60,  "max": 100},   # bpm
    "systolic_bp":      {"min": 90,  "max": 140},   # mmHg
    "diastolic_bp":     {"min": 60,  "max": 90},    # mmHg
    "spo2":             {"min": 95,  "max": 100},   # %
    "respiratory_rate": {"min": 12,  "max": 20},    # rpm
    "temperature":      {"min": 36.0,"max": 37.5},  # °C
}


class VitalsAnomalyDetector:
    """
    Detecta anomalias em séries temporais de sinais vitais
    usando Isolation Forest (não supervisionado).

    Formato esperado do CSV de entrada:
        timestamp, heart_rate, systolic_bp, diastolic_bp, spo2, respiratory_rate, temperature
    """

    def __init__(self, contamination: float = ANOMALY_THRESHOLD):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.scaler = StandardScaler()
        self.feature_columns = list(CLINICAL_RANGES.keys())

    def process_csv(self, csv_path: str, patient_id: str = "desconhecido") -> dict:
        """
        Lê CSV de sinais vitais, detecta anomalias e gera relatório.

        Args:
            csv_path: Caminho para o CSV com as séries temporais.
            patient_id: Identificador do paciente para o relatório.

        Returns:
            Relatório com timestamps anômalos e detalhes clínicos.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        logger.info(f"Carregado CSV: {csv_path.name} ({len(df)} registros)")

        available_features = [c for c in self.feature_columns if c in df.columns]
        if not available_features:
            raise ValueError("Nenhuma coluna de sinal vital reconhecida no CSV.")

        X = df[available_features].fillna(method="ffill").values
        X_scaled = self.scaler.fit_transform(X)

        scores = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        df["is_anomaly"] = scores == -1
        df["anomaly_score"] = anomaly_scores

        anomalous = df[df["is_anomaly"]].copy()
        logger.info(f"{len(anomalous)} anomalias detectadas de {len(df)} registros.")

        report = self._build_report(patient_id, csv_path.name, df, anomalous, available_features)
        self._save_report(report, patient_id)
        return report

    def _classify_severity(self, row: pd.Series, features: list) -> str:
        """Classifica severidade com base nos limites clínicos."""
        out_of_range = 0
        for feat in features:
            if feat in CLINICAL_RANGES and feat in row:
                val = row[feat]
                rng = CLINICAL_RANGES[feat]
                if val < rng["min"] or val > rng["max"]:
                    out_of_range += 1
        if out_of_range >= 3:
            return "crítica"
        elif out_of_range == 2:
            return "alta"
        elif out_of_range == 1:
            return "média"
        return "baixa"

    def _build_report(self, patient_id: str, source_file: str,
                      full_df: pd.DataFrame, anomalous: pd.DataFrame,
                      features: list) -> dict:

        anomaly_list = []
        for _, row in anomalous.iterrows():
            entry = {
                "timestamp": str(row.get("timestamp", "")),
                "anomaly_score": round(float(row["anomaly_score"]), 4),
                "severity": self._classify_severity(row, features),
                "vitals": {f: round(float(row[f]), 2) for f in features if f in row},
            }
            anomaly_list.append(entry)

        return {
            "patient_id": patient_id,
            "source_file": source_file,
            "generated_at": datetime.now().isoformat(),
            "total_records": len(full_df),
            "total_anomalies": len(anomalous),
            "anomaly_rate_pct": round(len(anomalous) / len(full_df) * 100, 2),
            "features_analyzed": features,
            "anomalies": anomaly_list,
            "alert": len(anomalous) > 0,
        }

    def _save_report(self, report: dict, patient_id: str):
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        safe_id = patient_id.replace(" ", "_")
        output_path = REPORTS_PATH / f"vitals_{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Relatório de vitais salvo em: {output_path}")
