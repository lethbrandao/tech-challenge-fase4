"""
Módulo 1 — Análise de Vídeo com YOLOv8
Detecta objetos e eventos anômalos em vídeos clínicos.
"""

import os
import cv2
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.5))
SAMPLE_FPS = int(os.getenv("VIDEO_SAMPLE_FPS", 2))
REPORTS_PATH = Path(os.getenv("REPORTS_PATH", "reports"))


class YOLOVideoAnalyzer:
    """
    Processa vídeos clínicos frame a frame usando YOLOv8,
    identificando objetos e sinalizando eventos fora do padrão.
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        logger.info(f"Carregando modelo YOLO: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def process_video(self, video_path: str) -> dict:
        """
        Processa um vídeo e retorna um relatório com as detecções por frame.

        Args:
            video_path: Caminho para o arquivo de vídeo.

        Returns:
            Dicionário com metadados e lista de eventos detectados.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / SAMPLE_FPS))

        events = []
        frame_idx = 0

        logger.info(f"Processando vídeo: {video_path.name} ({original_fps:.1f} fps)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / original_fps
                detections = self._analyze_frame(frame, timestamp_sec)
                if detections:
                    events.extend(detections)

            frame_idx += 1

        cap.release()
        logger.info(f"Processamento concluído. {len(events)} eventos detectados.")

        report = self._build_report(video_path.name, events)
        self._save_report(report, video_path.stem)
        return report

    def _analyze_frame(self, frame, timestamp_sec: float) -> list:
        """Roda inferência YOLO em um frame e retorna as detecções relevantes."""
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence >= self.confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    detections.append({
                        "timestamp_sec": round(timestamp_sec, 2),
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": box.xyxy[0].tolist(),
                    })

        return detections

    def _build_report(self, video_name: str, events: list) -> dict:
        """Monta o relatório final com resumo das detecções."""
        class_counts = {}
        for e in events:
            class_counts[e["class"]] = class_counts.get(e["class"], 0) + 1

        return {
            "video": video_name,
            "generated_at": datetime.now().isoformat(),
            "total_events": len(events),
            "summary": class_counts,
            "events": events,
        }

    def _save_report(self, report: dict, stem: str):
        """Salva o relatório em JSON na pasta reports/."""
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_PATH / f"video_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Relatório salvo em: {output_path}")
