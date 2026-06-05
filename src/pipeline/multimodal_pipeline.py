"""
Pipeline Multimodal
Orquestra os 3 módulos (vídeo, áudio, sinais vitais) em sequência,
passando os relatórios para o sistema de alertas.
"""

from loguru import logger
from src.video_analysis.yolo_analyzer import YOLOVideoAnalyzer
from src.video_analysis.pose_analyzer import PoseAnalyzer
from src.audio_analysis.speech_analyzer import SpeechAnalyzer
from src.anomaly_detection.vitals_detector import VitalsAnomalyDetector
from src.anomaly_detection.alert_system import AlertSystem


class MultimodalPipeline:
    """
    Ponto de entrada único para processar uma sessão clínica completa.
    Recebe caminhos opcionais para vídeo, áudio e CSV de vitais.
    """

    def __init__(self):
        self.alert_system = AlertSystem()
        logger.info("Pipeline multimodal inicializado.")

    def run(
        self,
        video_path: str | None = None,
        audio_path: str | None = None,
        vitals_csv_path: str | None = None,
        patient_id: str = "desconhecido",
    ) -> dict:
        """
        Executa os módulos disponíveis com base nos arquivos fornecidos.

        Args:
            video_path: Caminho para vídeo clínico (.mp4, .avi, etc.)
            audio_path: Caminho para áudio da consulta (.wav)
            vitals_csv_path: Caminho para CSV de sinais vitais
            patient_id: ID do paciente para rastreabilidade

        Returns:
            Dicionário com resultados de cada módulo e resumo de alertas.
        """
        results = {"patient_id": patient_id}

        # ── Módulo 1: Vídeo ───────────────────────────────────────────────────
        if video_path:
            logger.info("=== Módulo 1: Análise de Vídeo ===")
            try:
                yolo = YOLOVideoAnalyzer()
                video_report = yolo.process_video(video_path)
                self.alert_system.process_report(video_report, source="video")
                results["video_yolo"] = video_report

                pose = PoseAnalyzer()
                pose_report = pose.process_video(video_path)
                self.alert_system.process_report(pose_report, source="video")
                results["video_pose"] = pose_report
            except Exception as e:
                logger.error(f"Erro no módulo de vídeo: {e}")
                results["video_error"] = str(e)

        # ── Módulo 2: Áudio ───────────────────────────────────────────────────
        if audio_path:
            logger.info("=== Módulo 2: Análise de Áudio ===")
            try:
                speech = SpeechAnalyzer()
                audio_report = speech.process_audio(audio_path)
                self.alert_system.process_report(audio_report, source="audio")
                results["audio"] = audio_report
            except Exception as e:
                logger.error(f"Erro no módulo de áudio: {e}")
                results["audio_error"] = str(e)

        # ── Módulo 3: Sinais Vitais ───────────────────────────────────────────
        if vitals_csv_path:
            logger.info("=== Módulo 3: Detecção de Anomalias ===")
            try:
                detector = VitalsAnomalyDetector()
                vitals_report = detector.process_csv(vitals_csv_path, patient_id=patient_id)
                self.alert_system.process_report(vitals_report, source="vitals")
                results["vitals"] = vitals_report
            except Exception as e:
                logger.error(f"Erro no módulo de sinais vitais: {e}")
                results["vitals_error"] = str(e)

        # ── Consolidação de alertas ───────────────────────────────────────────
        results["alerts_summary"] = self.alert_system.get_summary()
        logger.info(f"Pipeline concluído. {results['alerts_summary']['total_alerts']} alerta(s) gerado(s).")
        return results
