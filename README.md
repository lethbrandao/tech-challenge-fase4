# Tech Challenge Fase 4 — IA para Devs (FIAP)

Sistema multimodal de monitoramento clínico com IA, integrando análise de vídeo, áudio e sinais vitais para detecção precoce de anomalias em ambiente hospitalar.

---

## Estrutura do projeto

```
tech-challenge-fase4/
├── src/
│   ├── video_analysis/
│   │   ├── yolo_analyzer.py       # Detecção de objetos com YOLOv8
│   │   └── pose_analyzer.py       # Análise postural com MediaPipe
│   ├── audio_analysis/
│   │   └── speech_analyzer.py     # Transcrição + análise de sentimento (Azure)
│   ├── anomaly_detection/
│   │   ├── vitals_detector.py     # Isolation Forest em séries temporais
│   │   └── alert_system.py        # Centralização e emissão de alertas
│   └── pipeline/
│       └── multimodal_pipeline.py # Orquestrador dos 3 módulos
├── data/
│   ├── raw/                        # Dados brutos (não versionados)
│   └── processed/                  # Dados processados (não versionados)
├── notebooks/                      # Exploração e demonstração
├── reports/                        # Relatórios JSON gerados automaticamente
├── tests/
├── .env.example                    # Template de variáveis de ambiente
├── .gitignore
└── requirements.txt
```

---

## Instalação

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd tech-challenge-fase4

# Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com suas chaves Azure
```

---

## Configuração do Azure

Você precisará de um recurso Azure com os seguintes serviços:

1. **Azure Cognitive Services — Speech** → gera `AZURE_SPEECH_KEY` e `AZURE_SPEECH_REGION`
2. **Azure Cognitive Services — Language (Text Analytics)** → gera `AZURE_TEXT_ANALYTICS_ENDPOINT` e `AZURE_TEXT_ANALYTICS_KEY`

Insira essas chaves no arquivo `.env` (nunca suba o `.env` para o repositório).

---

## Como usar

### Rodar o pipeline completo

```python
from src.pipeline.multimodal_pipeline import MultimodalPipeline

pipeline = MultimodalPipeline()
results = pipeline.run(
    video_path="data/raw/sessao_fisioterapia.mp4",
    audio_path="data/raw/consulta.wav",
    vitals_csv_path="data/raw/sinais_vitais.csv",
    patient_id="paciente_001",
)
print(results["alerts_summary"])
```

### Rodar módulos individualmente

```python
# Apenas vídeo
from src.video_analysis.yolo_analyzer import YOLOVideoAnalyzer
report = YOLOVideoAnalyzer().process_video("data/raw/video.mp4")

# Apenas áudio
from src.audio_analysis.speech_analyzer import SpeechAnalyzer
report = SpeechAnalyzer().process_audio("data/raw/consulta.wav")

# Apenas sinais vitais
from src.anomaly_detection.vitals_detector import VitalsAnomalyDetector
report = VitalsAnomalyDetector().process_csv("data/raw/vitais.csv", patient_id="p001")
```

### Formato do CSV de sinais vitais

```
timestamp,heart_rate,systolic_bp,diastolic_bp,spo2,respiratory_rate,temperature
2024-01-15 08:00:00,72,120,80,98,16,36.5
2024-01-15 08:05:00,75,122,81,97,17,36.6
...
```

Datasets recomendados: [PhysioNet](https://physionet.org/) | [Google AudioSet](https://research.google.com/audioset/)

---

## Módulos

### Módulo 1 — Análise de Vídeo
- **YOLOv8**: detecta objetos e eventos anômalos frame a frame
- **MediaPipe Pose**: monitora variações bruscas de postura
- Gera relatório JSON com timestamp, classe detectada e score de confiança

### Módulo 2 — Análise de Áudio
- **Azure Speech to Text**: transcreve áudios de consultas (pt-BR)
- **Azure Text Analytics**: analisa sentimento e detecta termos críticos
- Emite alerta automático se termos de risco forem identificados

### Módulo 3 — Detecção de Anomalias
- **Isolation Forest**: detecta outliers em séries temporais de sinais vitais
- Classifica severidade por faixas de referência clínica
- Integra com o sistema de alertas central

