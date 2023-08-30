"""
A simple anomaly detection task on the incoming stream data: 
(stream processing code using PySpark's Structured Streaming module)   
In this example, the code reads data from a Kafka topic, splits the incoming 
values into columns, converts them to double type, assembles them into a 
feature vector, and performs KMeans clustering for anomaly detection. Anomalies 
are identified by comparing the predicted cluster with the cluster center with 
the minimum distance.

Replace "localhost:9092" with your actual Kafka broker address and 
"your_topic_name" with the Kafka topic you want to read from. Also, adjust the 
schema and clustering parameters according to your data and use case.

Tip. Make sure you have Apache Spark and PySpark installed, and you'll need 
access to a Kafka broker with a topic to read from.

@author: khosravm
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Create a Spark session
spark = SparkSession.builder \
    .appName("AnomalyDetectionStream") \
    .getOrCreate()

# Define the schema for the incoming data
schema = StructType([
    StructField("feature1", DoubleType()),
    StructField("feature2", DoubleType()),
    # ... add more features as needed
])

# Read data from Kafka topic
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "your_topic_name") \
    .load()

# Convert the Kafka value column to string and split features
split_cols = kafka_df.value.cast("string").alias("value")
split_df = kafka_df.selectExpr("split(value, ',') as value").select("value.*")

# Convert columns to double and assemble into a feature vector
for col_name in schema.fieldNames():
    split_df = split_df.withColumn(col_name, col(col_name).cast("double"))
assembler    = VectorAssembler(inputCols=schema.fieldNames(), outputCol="features")
feature_df   = assembler.transform(split_df)

# Train a KMeans clustering model for anomaly detection
kmeans = KMeans(k=5, seed=1, featuresCol="features", predictionCol="cluster")
model  = kmeans.fit(feature_df)

# Perform anomaly detection based on cluster predictions
clustered_df = model.transform(feature_df)
anomaly_df   = clustered_df.withColumn("is_anomaly", (col("cluster") != model.clusterCenters().argmin()))

# Start the streaming query
query = anomaly_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
