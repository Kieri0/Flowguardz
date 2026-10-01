FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py index.html styles.css app.js model.pkl ./

ENV HOST=0.0.0.0
EXPOSE 8765
CMD ["python", "server.py"]
