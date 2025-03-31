FROM python:3.10-slim

# Cài gói hệ thống cần cho prophet
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libatlas-base-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và nâng pip + cài packages
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ app
COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]