FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY main.py .

# data and models live in a mounted volume, never in the image
ENV SMA_DATA_DIR=/data
VOLUME /data

ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "download"]