"""
Módulo 2 — Análise de Áudio
Transcreve áudios de consultas com Azure Speech to Text
e analisa sentimentos/termos críticos com Azure Text Analytics.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
TEXT_ANALYTICS_ENDPOINT = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
TEXT_ANALYTICS_KEY = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
AUDIO_LANGUAGE = os.getenv("AUDIO_LANGUAGE", "pt-BR")
REPORTS_PATH = Path(os.getenv("REPORTS_PATH", "reports"))

# Termos que devem disparar alerta imediato se detectados na transcrição
CRITICAL_TERMS = [
    "dor intensa", "não consigo respirar", "perda de consciência",
    "sangramento", "convulsão", "desmaio", "pressão muito alta",
    "dificuldade de engolir", "dormência", "paralisia",
]


class SpeechAnalyzer:
    """
    Pipeline de análise de áudio médico:
    1. Transcreve com Azure Speech to Text
    2. Analisa sentimento com Azure Text Analytics
    3. Detecta termos críticos e gera alerta se necessário
    """

    def __init__(self):
        self._validate_credentials()
        self.text_client = TextAnalyticsClient(
            endpoint=TEXT_ANALYTICS_ENDPOINT,
            credential=AzureKeyCredential(TEXT_ANALYTICS_KEY),
        )
        logger.info("SpeechAnalyzer inicializado.")

    def _validate_credentials(self):
        missing = [
            name for name, val in {
                "AZURE_SPEECH_KEY": SPEECH_KEY,
                "AZURE_TEXT_ANALYTICS_ENDPOINT": TEXT_ANALYTICS_ENDPOINT,
                "AZURE_TEXT_ANALYTICS_KEY": TEXT_ANALYTICS_KEY,
            }.items() if not val
        ]
        if missing:
            raise EnvironmentError(f"Variáveis de ambiente ausentes: {', '.join(missing)}")

    def process_audio(self, audio_path: str) -> dict:
        """
        Processa um arquivo de áudio e retorna relatório completo.

        Args:
            audio_path: Caminho para arquivo .wav (16kHz, mono recomendado).

        Returns:
            Dicionário com transcrição, sentimento, termos críticos e alertas.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

        logger.info(f"Transcrevendo: {audio_path.name}")
        transcription = self._transcribe(str(audio_path))

        logger.info("Analisando sentimento e entidades...")
        sentiment = self._analyze_sentiment(transcription)
        critical_hits = self._detect_critical_terms(transcription)

        report = self._build_report(audio_path.name, transcription, sentiment, critical_hits)
        self._save_report(report, audio_path.stem)
        return report

    def _transcribe(self, audio_path: str) -> str:
        """Usa Azure Speech SDK para transcrever o áudio."""
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.speech_recognition_language = AUDIO_LANGUAGE
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            logger.warning("Nenhuma fala reconhecida no áudio.")
            return ""
        else:
            logger.error(f"Erro na transcrição: {result.cancellation_details}")
            return ""

    def _analyze_sentiment(self, text: str) -> dict:
        """Analisa sentimento do texto transcrito via Azure Text Analytics."""
        if not text:
            return {"sentiment": "desconhecido", "scores": {}}

        response = self.text_client.analyze_sentiment([text], language="pt")[0]

        if response.is_error:
            logger.error(f"Erro no Text Analytics: {response.error}")
            return {"sentiment": "erro", "scores": {}}

        return {
            "sentiment": response.sentiment,
            "scores": {
                "positive": round(response.confidence_scores.positive, 3),
                "neutral": round(response.confidence_scores.neutral, 3),
                "negative": round(response.confidence_scores.negative, 3),
            },
        }

    def _detect_critical_terms(self, text: str) -> list:
        """Verifica presença de termos críticos na transcrição."""
        text_lower = text.lower()
        return [term for term in CRITICAL_TERMS if term in text_lower]

    def _build_report(self, audio_name: str, transcription: str, sentiment: dict, critical_hits: list) -> dict:
        alert = len(critical_hits) > 0 or sentiment.get("scores", {}).get("negative", 0) > 0.7

        return {
            "audio": audio_name,
            "generated_at": datetime.now().isoformat(),
            "transcription": transcription,
            "sentiment": sentiment,
            "critical_terms_detected": critical_hits,
            "alert": alert,
            "alert_reason": (
                f"Termos críticos: {critical_hits}" if critical_hits
                else ("Sentimento negativo elevado" if alert else None)
            ),
        }

    def _save_report(self, report: dict, stem: str):
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_PATH / f"audio_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Relatório de áudio salvo em: {output_path}")
