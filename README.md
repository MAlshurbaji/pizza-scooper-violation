# Pizza Store Scooper Violation Detection

![Framework](figures/overall_approach.png)

# Pizza Store Scooper Violation Detection

This project is a **real-time, microservices-based computer vision system** for monitoring hygiene compliance in a pizza store environment.  
It automatically detects scenarios where a worker **interacts with a protected ingredient container (ROI)** and **places ingredients on a pizza without using a scooper**, which is considered a hygiene violation.

The system processes **video files or live camera streams in real time**, using a modular pipeline of independent services for frame ingestion, object detection, temporal tracking, violation reasoning, and visualization. Each service operates independently and communicates asynchronously, making the system **scalable, maintainable, and production-ready**.

Violations are logged with visual evidence, stored in a database, and streamed live to a **web-based dashboard** that highlights detections, ROIs, and violation events.

- A **detailed, service-level technical documentation** is available in `\documentation.md`.  
- A brief demo video of the framework is available on [YouTube]().

> **Note:** This project is for **learning purposes only**.  
> The resulting models are not intended to be competitive.

**Release date:** 04/Jan/2026

## ✨Features

- Object detection: **hand, scooper, pizza, person**
- ROI-based monitoring (ingredient containers)
- Temporal logic (hand → ROI → pizza → scooper check)
- False-positive reduction (persistence, ignore windows, future checks)
- Saves:
  - Violation frames
  - Violation metadata.
- Live UI:
  - Bounding boxes
  - ROIs
  - Violation thumbnails
  - Clickable short video context around violations
- Fully **Dockerized microservices architecture**

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

## 📁 Project Structure

```
pizza-scooper-violation/
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
├── tools/
│   ├── roi/
│   ├── select_roi.py/
│   └── save_annotated_intervals.py
├── docker-compose.yml
├── .env
├── documentation.md
└── README.md
```

## 📥 Data & Model Downloads (Required)

Before running the system, download the following resources:

- **Pretrained Detection Model (YOLO)**  
  Download from https://drive.google.com/drive/folders/1S_WeBU-o3QRRAbn9HCFHSt-3uuPtsQ8K  

- **Unannotated Videos**  
  Download from https://drive.google.com/drive/folders/1lbYQgANVBJ7IIz0uNgnhZt5gMV0PpeaK  

- You can draw the Region-Of-Interest (ROI) using `/tools/select_roi.py`, which then will save the ROIs json file in `configs/rois.json`, and as images in `/tools/roi/`.

#### Place required files:
- Videos → `data/videos/`
- Model → `models/yolo12m-v2.pt`
- ROIs → `configs/rois.json`
  
# 🚀 Option 1 — Run with Docker (Recommended)

Docker provides reproducibility, isolation, and one-command startup for the entire system.

### Prerequisites
- Docker Desktop
- RabbitMQ & Erlang (installed locally)
- WSL 2
- CUDA

### 1️⃣ Run the entire framework

```bash
cd '\pizza-scooper-violation'
docker compose up -d    # This will take a couple of minutes (for the first time).
```

### 2️⃣ Access services in a web browser

- **Frontend UI**: http://localhost:3000
- **Streaming API / WebSocket**: http://localhost:8003
- **RabbitMQ UI**: http://localhost:15672  
  user: `guest` | pass: `guest`

### 3️⃣ To Stop the framework

```bash
docker compose stop     # stop containers (keep them)
docker compose down     # stop & remove containers (keep images)
```

# 🖥️ Option 2 — Run without Docker (Local Python)

### Prerequisites
- Python 3.10+
- Conda
- RabbitMQ & Erlang (installed locally)
- CUDA

### 1️⃣ Create an environment and install dependencies

```bash
conda create -n pizza python=3.10
conda activate pizza
cd '\pizza-scooper-violation'

pip install pika opencv-python-headless numpy fastapi uvicorn websockets ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # GPU support
```

### 2️⃣ Start RabbitMQ
In `RabbitMQ Command Prompt (sbin)` run:
```bat
rabbitmq-plugins enable rabbitmq_management
```

### 3️⃣ Run services (each in a separate terminal)

```bash
python services/frame_reader/app.py   # in the 1st terminal
python services/detection/app.py      # in a 2nd terminal
python services/tracker/app.py        # in a 3rd terminal
python services/streaming/app.py      # in a 4th terminal

cd '\services\frontend'               # in a 5th terminal
python -m http.server 3000            # in the same 5th terminal
```

### 4️⃣ Access services in a web browser

- **Frontend UI**: http://localhost:3000
- **Streaming API / WebSocket**: http://localhost:8003
- **RabbitMQ UI**: http://localhost:15672  
  user: `guest` | pass: `guest`

## 🧠 Violation Logic (Summary)

A violation is recorded when:
1. A hand enters a defined ROI
2. The same hand later touches a pizza
3. No scooper is detected near the hand
4. Conditions persist across multiple frames
5. No scooper appears within a future grace window
