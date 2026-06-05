"""
Sistema de Alertas
Centraliza e consolida os alertas gerados pelos 3 módulos,
exibindo-os no terminal e (futuramente) podendo enviar via webhook/email.
"""

import json
from datetime import datetime
from pathlib import Path
from loguru import logger


class AlertSystem:
    """
    Recebe relatórios dos módulos de vídeo, áudio e sinais vitais
    e emite alertas consolidados para a equipe médica.
    """

    def __init__(self, log_path: str = "reports/alerts.log"):
        self.alerts = []
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def process_report(self, report: dict, source: str):
        """
        Avalia um relatório e gera alerta se necessário.

        Args:
            report: Dicionário retornado por qualquer um dos analisadores.
            source: 'video' | 'audio' | 'vitals'
        """
        if not report.get("alert", False):
            return

        alert = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "severity": self._determine_severity(report, source),
            "message": self._build_message(report, source),
            "details": report,
        }

        self.alerts.append(alert)
        self._log_alert(alert)

    def _determine_severity(self, report: dict, source: str) -> str:
        if source == "vitals":
            severities = [a.get("severity", "baixa") for a in report.get("anomalies", [])]
            if "crítica" in severities:
                return "CRÍTICA"
            elif "alta" in severities:
                return "ALTA"
            return "MÉDIA"
        elif source == "audio":
            if report.get("critical_terms_detected"):
                return "ALTA"
            return "MÉDIA"
        else:  # video
            return "MÉDIA"

    def _build_message(self, report: dict, source: str) -> str:
        if source == "vitals":
            n = report.get("total_anomalies", 0)
            return f"Paciente {report.get('patient_id', '?')}: {n} anomalia(s) em sinais vitais detectada(s)."
        elif source == "audio":
            terms = report.get("critical_terms_detected", [])
            if terms:
                return f"Termos críticos detectados na consulta: {', '.join(terms)}."
            return "Sentimento negativo elevado detectado na consulta."
        else:
            n = report.get("total_events", 0)
            return f"Vídeo '{report.get('video', '?')}': {n} evento(s) anômalo(s) detectado(s)."

    def _log_alert(self, alert: dict):
        severity = alert["severity"]
        message = alert["message"]
        logger.warning(f"[ALERTA {severity}] {message}")

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")

    def get_summary(self) -> dict:
        """Retorna resumo de todos os alertas da sessão."""
        return {
            "total_alerts": len(self.alerts),
            "by_severity": {
                "CRÍTICA": sum(1 for a in self.alerts if a["severity"] == "CRÍTICA"),
                "ALTA": sum(1 for a in self.alerts if a["severity"] == "ALTA"),
                "MÉDIA": sum(1 for a in self.alerts if a["severity"] == "MÉDIA"),
            },
            "alerts": self.alerts,
        }
