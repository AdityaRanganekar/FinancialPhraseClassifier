FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only PyTorch to prevent memory crashes and shrink image size
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]