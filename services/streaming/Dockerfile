FROM python:3.10-slim
WORKDIR /app
COPY . /app
# prefer full uvicorn extras or at minimum install a websocket backend
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pika opencv-python-headless numpy
# OR minimal:
# RUN pip install --no-cache-dir fastapi uvicorn websockets pika opencv-python-headless numpy
ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]
