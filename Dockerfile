FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Instala dependências do sistema necessárias para processamento de áudio (librosa/soundfile) e compilação
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o arquivo de requisitos e instala as bibliotecas Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# O comando padrão quando o container iniciar
ENTRYPOINT ["python", "extract_embs/extract_f0.py"]
# Argumentos padrão: base_dir como raiz onde os volumes serão montados
CMD ["-b", "/", "-i", "input", "-o", "output"]
