FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-build.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-build.txt pyinstaller

COPY . .

RUN pyinstaller --onedir --add-data "config.yaml:." \
    --add-data "modules/voices:modules/voices" \
    --collect-all vispy \
    --collect-all whisper \
    --collect-all faster_whisper \
    --collect-all uvicorn \
    --collect-all PyQt5 \
    --collect-all sudachidict_core \
    --collect-all gruut_lang_de \
    --collect-all gruut_lang_es \
    --collect-all gruut_lang_fr \
    --collect-all piper \
    main.py

COPY run.sh dist/main/run.sh
COPY install.sh dist/main/install.sh
RUN chmod +x dist/main/run.sh dist/main/install.sh

CMD ["true"]