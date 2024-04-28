FROM python:3.8-slim-buster
WORKDIR /app
ADD . /app

# Install Java and wget
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk-headless wget

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2 /opt/spark && \
    rm spark-3.1.2-bin-hadoop3.2.tgz

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Use ENTRYPOINT to specify the executable
ENTRYPOINT ["spark-submit", "--master", "local[*]", "--class", "com.example.SparkMlWineQualityPred", "spark-ml-wine-quality-1.0-SNAPSHOT.jar"]

