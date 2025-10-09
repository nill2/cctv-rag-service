# 🧠 Face Search Engine (Offline)

A lightweight, privacy-first face search and photo analysis service.
Runs 100% locally — no cloud APIs, no OpenAI, no data sharing.

---

## 🚀 Features

- Detects faces and stores embeddings (from your main `face_detection` app)
- Finds all photos containing a specific person (e.g., “Danil”)
- Detects **unknown visitors** — people not in your known face set
- 100% local MongoDB + NumPy + cosine similarity

---

## 🧰 Stack

- Python 3.11
- FastAPI
- MongoDB (local or Atlas)
- Docker-ready

---

## ⚙️ Setup

1. Clone this repo:

```bash
git clone https://github.com/
<your-user>/face_search_engine.git
```

2. Create `.env` file:

```bash
MONGO_HOST=mongodb://localhost:27017
MONGO_DB=nill-home
FACE_COLLECTION=nill-faces
KNOWN_FACES_COLLECTION=nill-known-faces
```

3. Run via Docker:

```bash
docker-compose up --build
```

4. Access API:

```bash
http://localhost:8000/docs
```

---

## 🧩 Example Queries

**Find all photos with Danil:**

```md
GET /search/known?name=Danil
```
