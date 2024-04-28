## Spark ML Wine Quality Prediction

### Step 1: Get AWS Creds
1. Initialize the lab.
2. Select "AWS Details."
3. Download the SSH .pem file for EC2.

### Step 2: Data Upload to S3
1. Open S3 in the AWS console.
2. Create a bucket: "spark-ml-wine-quality."
3. Add `TrainingDataset.csv` and `ValidationDataset.csv`.

### Step 3: Start EMR Cluster
1. Open AWS Console, choose "EMR" and then "create cluster."
2. Add Spark, set core instances to 4, disable auto-termination, and pick an SSH key.
3. Use default settings for the rest and Launch the cluster.
4. Update the master's security group accesses to allow for SSH access.

### Step 4: Configuring the PEM File
1. open a terminal.
2. Relocate the .pem file and update permissions:
   ```bash
   chmod 400 /path/to/your_key.pem
   ```

### Step 5: Jar File
1. Using a terminal, transfer the jar file to the master node:
   ```bash
   scp -i /path/to/your_key.pem /path/to/file/spark-ml-wine-quality-1.0-SNAPSHOT.jar hadoop@<master_instance_public_ip>:/home/hadoop
   ```
   update the above command with actual file paths.

### Step 6: Master Instance SSH Connection
1. Connect via SSH:
   ```bash
   ssh -i /path/to/your_key.pem hadoop@<master_instance_public_ip>
   ```

### Step 7: Model Training
Run the Spark training script:
```bash
spark-submit --class com.example.SparkMlWineQualityTrain --master yarn spark-ml-wine-quality-1.0-SNAPSHOT.jar
```
- The model is trained and saved at `s3://ml-spark-ml-wine-quality/best_model`.

### Step 8: Prediction Script without Docker
Run the Spark prediction script:
```bash
spark-submit --class com.example.SparkMlWineQualityPred --master yarn spark-ml-wine-quality-1.0-SNAPSHOT.jar s3://spark-ml-wine-quality/best_model/ s3://spark-ml-wine-quality/ValidationDataset.csv 
```

### Step 9: Docker Setup for Prediction
1. Docker login:
   ```bash
   docker login
   ```
2. Move to the repository folder.
3. Create a Docker image:
   ```bash
   docker build -t <docker-hub-username>/spark-ml-wine-quality .
   ```
   Replace with your Docker Hub username.
4. Upload the image:
   ```bash
   docker push <docker-hub-username>/spark-ml-wine-quality
   ```

### Step 10: Prediction on Docker
1. Launch an EC2 instance.
2. Download the best model folder from S3.
3. copy the model and `ValidationDataset.csv` to EC2.
4. SSH into EC2:
   ```bash
   ssh -i /path/to/your_key.pem ec2_user@master_instance_public_dns
   ```
5. Download the Docker image:
   ```bash
   docker pull <docker-hub-username>/spark-ml-wine-quality
   ```
6. Execute prediction in Docker:
   ```bash
   docker run -v $(pwd):/inputs <docker-hub-username>/spark-ml-wine-quality /inputs/best_model /inputs/ValidationDataset.csv
   ```
