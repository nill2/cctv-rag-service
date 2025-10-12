FROM python:3.11-slim

WORKDIR /app
COPY . .

# Remove Windows-only dependency (pywin32) before installation
RUN grep -v "pywin32" requirements.txt > requirements-clean.txt

# Install all requirements (clean version for Linux)
RUN pip install --no-cache-dir -r requirements-clean.txt

# (Optional) Install original file if needed locally
# RUN pip install --no-cache-dir -r requirements.txt

# Expose both ports (main and alternate)
EXPOSE 8000
EXPOSE 8001

# Launch FastAPI app on port 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
