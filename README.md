# 🧠 CCTV RAG Service

A lightweight, privacy-first **face search and retrieval engine**.
Runs fully **offline** — no OpenAI, no cloud APIs, no external data sharing.

---

## 🚀 Features

- Searches for **known faces** stored in MongoDB
- Finds photos containing **specific people** (e.g., "Danil")
- Finds photos of **unknown visitors** (faces with no match)
- Supports **photo-based similarity search**
- Returns **latest CCTV entry** and **all detected photos**
- 100% local: uses **RAG-style vector search** with deterministic embeddings
- Ready for containerized deployment with **Docker**

---

## 🧠 Technologies

| Component                                | Purpose                                                               |
| ---------------------------------------- | --------------------------------------------------------------------- |
| **FastAPI**                              | Provides the REST API for search and similarity queries               |
| **MongoDB**                              | Stores face metadata and embeddings                                   |
| **NumPy + scikit-learn**                 | Performs vector similarity (cosine similarity)                        |
| **RAG (Retrieval-Augmented Generation)** | Used conceptually for local retrieval of face embeddings              |
| **MCP (Modular Compute Pipeline)**       | Separates data collection, embedding, and search for flexible scaling |

---

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

```
MONGO_HOST=
MONGO_PORT=
MONGO_DB=
MONGO_COLLECTION=
FACE_COLLECTION=
KNOWN_FACES_COLLECTION=
FTP_HOST=
FTP_PORT=
FTP_USER=
FTP_PASS=
FACES_HISTORY=
```

---

## 🐳 Run with Docker

```bash
docker-compose up --build
```

## 🧩 API Overview

The service provides REST endpoints for both **database-based face search** and **embedding similarity search**.

---

### 1. GET /health

**Description:**
Simple health check to confirm that the service is up and connected to MongoDB.

**Returns:**

```text
{ "status": "ok" }
```

---

### 2. POST /faces/known

**Description:**
Find all photos containing one or more known persons.

**Body (JSON):**

```text
{
  "names": ["Danil", "Alex"]
}
```

**Returns:**

```text
{
  "results": [...],
  "skipped_names": ["Unknown Person"]
}
```

---

### 3. GET /faces/unknown

**Description:**
Find all photos containing unknown faces — i.e., faces that were not matched to any known person.

**Returns:**

```text
{ "results": [...] }
```

---

### 4. POST /faces/search

**Description:**
Upload a photo and search for visually similar faces stored in the database.
Performs deterministic embedding generation and cosine similarity search.

**Body (form-data):**

- `file`: binary image
- `threshold`: optional float (default = 0.8)

**Returns:**

```text
{
  "results": [...],
  "threshold": 0.8
}
```

---

### 5. GET /saved_known_faces

**Description:**
Return all known faces from the `known_faces_collection`.

**Returns:**

```text
{ "results": [...] }
```

---

### 6. GET /photos_detected_faces

**Description:**
Return all photos that contain detected faces (from the `photos` collection).

**Returns:**

```text
{ "results": [...] }
```

---

### 7. GET /current_cctv

**Description:**
Return the most recent CCTV entry from the `photos` collection.

**Returns:**

```text
{ "result": {...} }
```

---

## 🧪 Example Workflow

1. Your main CCTV face detection app inserts detected faces into MongoDB.
2. The RAG service connects to that same database.
3. You can now:
   - Query known faces (`POST /faces/known`)
   - Detect unknowns (`GET /faces/unknown`)
   - Upload a photo to find matches (`POST /faces/search`)
   - Get the latest CCTV frame (`GET /current_cctv`)
   - List all detected photos (`GET /photos_detected_faces`)

---

## 🧱 Project Structure

```text
app/
├── db.py                # MongoDB connection helpers
├── face_search.py       # Known/unknown face search logic
├── rag_engine.py        # Photo-based similarity search
├── utils.py             # JSON serialization and helpers
main.py                  # FastAPI entrypoint
```

---

## 🔒 Privacy Note

All computation and data storage are fully local.
No API calls, telemetry, or data sharing occur.
This makes the service suitable for **private CCTV, home automation, and offline analytics** use cases.

---

## 📜 License

MIT License © 2025 [nill2](https://github.com/nill2)
