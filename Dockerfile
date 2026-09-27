FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src ./src
# trained weights are not in the image: mount them at run time
#   docker build -t multimodal-moderation .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models" multimodal-moderation
EXPOSE 8000
HEALTHCHECK CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health').raise_for_status()"
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
