"""
Módulo 1 — Análise Postural com MediaPipe
Detecta padrões anômalos de postura/movimento em vídeos clínicos.
Usa MediaPipe como alternativa open-source ao OpenPose.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

SAMPLE_FPS = int(os.getenv("VIDEO_SAMPLE_FPS", 2))
REPORTS_PATH = Path(os.getenv("REPORTS_PATH", "reports"))

# Limiar de variação angular para considerar movimento anômalo (graus)
ANGLE_ANOMALY_THRESHOLD = 45.0


class PoseAnalyzer:
    """
    Analisa postura e movimentos de pacientes em vídeos clínicos
    usando MediaPipe Pose (equivalente funcional ao OpenPose).
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("PoseAnalyzer inicializado com MediaPipe Pose.")

    def process_video(self, video_path: str) -> dict:
        """
        Processa vídeo e detecta anomalias posturais por frame.

        Args:
            video_path: Caminho para o arquivo de vídeo.

        Returns:
            Relatório com eventos posturais anômalos detectados.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / SAMPLE_FPS))

        anomalies = []
        frame_idx = 0
        prev_landmarks = None

        logger.info(f"Analisando postura em: {video_path.name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / original_fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = self._extract_landmarks(results.pose_landmarks)

                    if prev_landmarks is not None:
                        anomaly = self._detect_anomaly(landmarks, prev_landmarks, timestamp_sec)
                        if anomaly:
                            anomalies.append(anomaly)

                    prev_landmarks = landmarks

            frame_idx += 1

        cap.release()
        logger.info(f"Análise postural concluída. {len(anomalies)} anomalias detectadas.")

        report = self._build_report(video_path.name, anomalies)
        self._save_report(report, video_path.stem)
        return report

    def _extract_landmarks(self, pose_landmarks) -> np.ndarray:
        """Converte landmarks do MediaPipe para array numpy."""
        return np.array([
            [lm.x, lm.y, lm.z]
            for lm in pose_landmarks.landmark
        ])

    def _detect_anomaly(self, current: np.ndarray, previous: np.ndarray, timestamp: float) -> dict | None:
        """
        Compara landmarks consecutivos e sinaliza variações bruscas.
        Retorna dicionário de anomalia ou None.
        """
        delta = np.linalg.norm(current - previous, axis=1)
        max_delta_idx = int(np.argmax(delta))
        max_delta = float(delta[max_delta_idx])

        if max_delta > (ANGLE_ANOMALY_THRESHOLD / 100):  # normalizado 0-1
            landmark_name = self.mp_pose.PoseLandmark(max_delta_idx).name
            return {
                "timestamp_sec": round(timestamp, 2),
                "landmark": landmark_name,
                "movement_magnitude": round(max_delta, 4),
                "severity": "alta" if max_delta > 0.15 else "média",
            }
        return None

    def _build_report(self, video_name: str, anomalies: list) -> dict:
        return {
            "video": video_name,
            "generated_at": datetime.now().isoformat(),
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
        }

    def _save_report(self, report: dict, stem: str):
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_PATH / f"pose_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Relatório postural salvo em: {output_path}")
