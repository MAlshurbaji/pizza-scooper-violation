# pizza-scooper-violation
Real-time computer vision system that detects scooper-usage violations during pizza preparation, with live visualization and Dockerized microservices.

This is a **microservices-based computer vision system** designed to monitor hygiene compliance in a pizza store.
It detects cases where a worker **takes ingredients from a container (ROI)** and **places them on a pizza without using a scooper**.

The system supports **video files or live streams**, performs **real-time detection and tracking**, logs violations, and displays results in a **web-based dashboard**.

---

## ✨Features

- Object detection: **hand, scooper, pizza, person**
- ROI-based monitoring (ingredient containers)
- Temporal logic (hand → ROI → pizza → scooper check)
- False-positive reduction (persistence, ignore windows, future checks)
- Saves:
  - Violation frames
  - Violation metadata (SQLite)
- Live UI:
  - Bounding boxes
  - ROIs
  - Violation thumbnails
  - Clickable short video context around violations
- Fully **Dockerized microservices architecture**

---

## 🧱 Architecture

```
Frame Reader  →  RabbitMQ  →  Detection (YOLO, GPU/CPU)
                                   ↓
                              Tracker / Violation
                                   ↓
                           Streaming API + WebSocket
                                   ↓
                                Frontend UI
```

---

## 📁 Project Structure

```
Eagle Vision/
├── services/
│   ├── frame_reader/
│   ├── detection/
│   ├── tracker/
│   ├── streaming/
│   └── frontend/
├── data/
│   ├── videos/
│   ├── debug_detections/
│   ├── violations/
│   └── violations.db
├── models/
│   └── yolo12m-v2.pt
├── configs/
│   └── rois.json
├── docker-compose.yml
├── .env.example
└── README.md
```

---

# 🚀 Option 1 — Run with Docker (Recommended)

Docker provides reproducibility, isolation, and one-command startup for the entire system.

### Prerequisites
- Docker Desktop
- WSL 2
- NVIDIA GPU + drivers for GPU inference

---

### 1️⃣ Setup environment file

```bash
cd '\scooper-violation'
docker compose up -d
```

---

### 2️⃣ Place required files

- Videos → `data/videos/`
- Model → `models/yolo12m-v2.pt`
- ROIs → `configs/rois.json`

---

### 3️⃣ Run the entire framework

```bash
docker compose up -d
```

---

### 4️⃣ Access services

- **Frontend UI**: http://localhost:3000
- **Streaming API / WebSocket**: http://localhost:8003
- **RabbitMQ UI**: http://localhost:15672  
  user: `guest` | pass: `guest`

---

### 5️⃣ Stop the framework

```bash
docker compose stop     # stop containers (keep them)
docker compose down     # stop & remove containers (keep images)
```

---

# 🖥️ Option 2 — Run without Docker (Local Python)

---

### Prerequisites
- Python 3.10+
- Conda
- RabbitMQ (installed locally)
- CUDA

---

### 1️⃣ Create and activate environment

```bash
conda create -n pizza python=3.10
conda activate pizza
```

---

### 2️⃣ Install dependencies

```bash
pip install pika opencv-python-headless numpy fastapi uvicorn websockets ultralytics
```

GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 3️⃣ Start RabbitMQ

```bat
rabbitmq-plugins enable rabbitmq_management
```

---

### 4️⃣ Run services (each in a separate terminal)

```bash
python services/frame_reader/app.py
python services/detection/app.py
python services/tracker/app.py
python services/streaming/app.py
```

---

## 🧠 Violation Logic (Summary)

A violation is recorded when:
1. A hand enters a defined ROI
2. The same hand later touches a pizza
3. No scooper is detected near the hand
4. Conditions persist across multiple frames
5. No scooper appears within a future grace window

---

## 📦 Outputs

- `data/debug_detections/`
- `data/violations/`
- `data/violations.db`

---

## 📄 License

For academic and demonstration purposes.
