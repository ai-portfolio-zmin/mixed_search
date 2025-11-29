FROM python:3.10-slim

# Install Java (default JDK, headless) for Pyserini / pyjnius
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
 && rm -rf /var/lib/apt/lists/*

# Use the default JDK location
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV JVM_PATH=/usr/lib/jvm/default-java/lib/server/libjvm.so
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]