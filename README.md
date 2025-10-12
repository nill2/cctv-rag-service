# 🧠 CCTV RAG Service

A lightweight, privacy-first **face search and retrieval engine**.
Runs fully **offline** — no OpenAI, no cloud APIs, no external data sharing.

---

## 🚀 Features

- Searches for **known faces** stored in MongoDB
- Finds photos containing **specific people** (e.g., "Danil")
- Finds photos of **unknown visitors** (faces with no match)
- Supports **photo-based similarity search**
- 100% local: uses **RAG-style vector search** with deterministic embeddings
- Ready for containerized deployment with **Docker**

---

## 🧠 Technologies

| Component                                | Purpose                                                                                                  |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **FastAPI**                              | Provides the REST API for search and similarity queries                                                  |
| **MongoDB**                              | Stores face metadata and embeddings                                                                      |
| **NumPy + scikit-learn**                 | Performs vector similarity (cosine similarity)                                                           |
| **RAG (Retrieval-Augmented Generation)** | Used conceptually for local retrieval of face embeddings — “retrieve” most similar items before analysis |
| **MCP (Modular Compute Pipeline)**       | Separates data collection, embedding, and search into independent modules for flexible scaling           |

### Why RAG + MCP?

This project uses **RAG principles** to perform efficient **local retrieval** of embeddings — similar to how RAG retrieves relevant context for LLMs, but applied to **visual data**.
Combined with **MCP (Modular Compute Pipeline)** design, it allows clear separation of:

- Face data ingestion
- Embedding generation and storage
- Search and ranking by similarity

That makes it modular, maintainable, and ready for expansion into more advanced pipelines (e.g., integrating re-identification or clustering models later).

---

## ⚙️ Environment Variables

These environment variables are required for the service to run:

`MONGO_HOST=
MONGO_PORT=
MONGO_DB=
MONGO_COLLECTION=
FACE_COLLECTION=
FTP_HOST=
FTP_PORT=
FTP_USER=
FTP_PASS=
FACES_HISTORY=`

---

## 🐳 Run with Docker

`docker-compose up --build`

Then access the API docs in your browser at:

`http://localhost:8000/docs`

---

## 🧩 API Overview

The service provides REST endpoints for both **database-based face search** and **embedding similarity search**.

---

### 1. **GET /search/known**

**Description:**
Find all photos that contain a person with the given name.

**Query parameters:**

- `name` _(string, required)_ — Name of the person to search for.

**Returns:**

- `200 OK`: List of MongoDB documents representing photos where the person appears.
  Each document includes metadata such as file name, timestamp, and recognized faces.

---

### 2. **GET /search/unknown**

**Description:**
Find all photos containing **unknown faces** — i.e., faces that were not matched to any known person.

**Returns:**

- `200 OK`: List of MongoDB documents where `"matched_persons": []`.

---

### 3. **POST /search/photo**

**Description:**
Upload a photo and search for visually similar faces stored in the database.
Performs deterministic embedding generation (no model download needed) and cosine similarity search.

**Body:**

- `file` _(binary image)_ — The uploaded photo file.
- `threshold` _(float, optional, default=0.8)_ — Minimum similarity required to include a match.

**Returns:**

- `200 OK`: List of documents from the faces collection that are similar to the uploaded photo.
  Each entry includes:
  - `_id` — Document ID
  - `similarity` — Float score between 0 and 1
  - Additional face metadata from the database

---

### 4. **GET /health**

**Description:**
Simple health check to confirm that the service is up and connected to MongoDB.

**Returns:**

- `200 OK`: `{ "status": "ok" }`

---

## 🧪 Example Workflow

1. Your main CCTV face detection app inserts detected faces into MongoDB.
2. The RAG service connects to that same database.
3. You can now:
   - Query known faces (`/search/known?name=...`)
   - Detect unknowns (`/search/unknown`)
   - Upload a photo to find matches (`/search/photo`)

---

## 🧱 Project Structure

`app/
├── api/                 # FastAPI route handlers
├── db.py                # MongoDB connection helpers
├── face_search.py       # Known/unknown face search logic
├── rag_engine.py        # Photo-based similarity search
├── utils.py             # JSON serialization and helpers
main.py              # FastAPI entrypoint`

---

## 🔒 Privacy Note

All computation and data storage are fully local.
No API calls, telemetry, or data sharing occur.
This makes the service suitable for **private CCTV, home automation, and offline analytics** use cases.

---

## 📜 License

MIT License © 2025 [nill2](https://github.com/nill2)
