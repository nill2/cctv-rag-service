FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 8001

# Run FastAPI on port 8001 instead
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
