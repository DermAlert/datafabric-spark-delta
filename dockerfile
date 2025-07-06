FROM python:3.9-slim

# Install Java (OpenJDK 17) and other dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       openjdk-17-jdk \
       curl \
       wget \
       vim \
       procps \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME so PySpark can find Java
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Include the Delta Core JAR and Hadoop AWS JARs by specifying --packages in PYSPARK_SUBMIT_ARGS
ENV PYSPARK_SUBMIT_ARGS="--packages io.delta:delta-core_2.12:2.4.0,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 pyspark-shell"

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyspark \
    delta-spark==2.4.0 \
    minio \
    pandas \
    pyarrow \
    python-dotenv

# Environment variables - use minio container name for internal networking
ENV MINIO_ENDPOINT="minio:9000"
ENV MINIO_ACCESS_KEY="minio"
ENV MINIO_SECRET_KEY="minio123"
ENV BUCKET_NAME="isic-parquet-p"
ENV DELTA_PATH_PREFIX="delta"

# Copy your PySpark script into the container
COPY pyspark_query.py /app/pyspark_query.py

WORKDIR /app

# Run with list collections to test configuration
CMD ["tail", "-f", "/dev/null"]
# CMD ["python", "pyspark_query.py", "--list-collections"]


