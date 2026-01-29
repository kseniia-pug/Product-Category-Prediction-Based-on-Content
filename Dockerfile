FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
#COPY artifacts/ ./artifacts/  # Если артефакты загружаются локально
CMD ["python", "src/main.py"]