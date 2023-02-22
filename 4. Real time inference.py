# Databricks notebook source
# MAGIC %md 
# MAGIC ## Real time risk prediciton with A/B testing
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC In this notebook we will do live inference using the two models trained previously (models A and B) that are registered on MLflow. The streaming data will be obtained from the Delta table risk_stream_source
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_3.png?raw=true" width="1000"/>
# MAGIC 
# MAGIC 
# MAGIC When the streaming data arrives, model A will make predictions on 50% of the rows and the other 50% will be used by model B. The predictions will be writen to another Delta table. This table will later on be used to compute the quality metrics using the ground truth, and we will display it in Databricks SQL. As an alternative, if we wanted to connect to resources in an operational context, we could for example send the predictions to a Kafka server.
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/abtesting.png?raw=true"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook setup
# MAGIC Run the following cell to prepare the resources for this notebook

# COMMAND ----------

# MAGIC %run ./_resources/03_helper_inference

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the models
# MAGIC We can load the models using the MLflow API

# COMMAND ----------

import pyspark.sql.functions as F

model_a = mlflow.spark.load_model(
  model_uri=f"models:/{model_name}/{model_a_version}" # Logistic regression model
)

model_b = mlflow.spark.load_model(
  model_uri=f"models:/{model_name}/{model_b_version}" # Gradient boosting
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Read the streaming data

# COMMAND ----------

df = (
  spark
  .readStream
  .format("delta")
  .table("{}.risk_stream_source".format(dbName))
  .withColumn("timestamp", F.unix_timestamp(F.current_timestamp()))
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC If we wanted to read a stream from Kafka instead, the code would be very similar:
# MAGIC ~~~
# MAGIC df = (
# MAGIC   spark 
# MAGIC   .readStream 
# MAGIC   .format("kafka") 
# MAGIC   .option("kafka.bootstrap.servers", "host1:port1,host2:port2") 
# MAGIC   .option("subscribe", "topic1") 
# MAGIC   .load()
# MAGIC )
# MAGIC ~~~
# MAGIC 
# MAGIC More info at https://docs.databricks.com/structured-streaming/kafka.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Divide the incoming credit requests into two equally sized groups: A and B
# MAGIC We will use a random number generator to make the split. 50% of the request will be made with model B and the remaining 50% with model A.

# COMMAND ----------

b_model_fraction = 0.5

# COMMAND ----------

import pandas as pd
import numpy as np

df_with_split = (
  df
  .withColumn("random_number", F.rand())
  .withColumn("risk", F.lit("good"))# this "risk" is a dummy value because it is needed in the pipeline, it will be ignored though
) 

df_a = (
  df_with_split
  .where(F.col("random_number") >= b_model_fraction)
  .withColumn("group", F.lit("A"))
)

df_b = (
  df_with_split
  .where(F.col("random_number") < b_model_fraction)
  .withColumn("group", F.lit("B"))
)

display(df_a.union(df_b))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions with the two models
# MAGIC - Model A will make predictions on the data classified in group A
# MAGIC - Model B will to the same for the data in group B

# COMMAND ----------

cols_keep = ["group", "id", "prediction", "probability", "timestamp"]

df_pred_a = (
  model_a
  .transform(df_a)
  .select(cols_keep)
)

df_pred_b = (
  model_b
  .transform(df_b)
  .select(cols_keep)
)

df_pred = df_pred_a.union(df_pred_b)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to a Delta Table while streaming

# COMMAND ----------

(
  df_pred
  .writeStream
  .format("delta")
  .option("checkpointLocation", "/FileStore/tmp/streaming_ckpnt_risk_demo_{}".format(dbName))
  .table("{}.risk_stream_predictions".format(dbName))
)

# COMMAND ----------

# MAGIC %md
# MAGIC If we wanted to write to a stream from Kafka instead, the code would be very similar:
# MAGIC ~~~
# MAGIC (
# MAGIC   df 
# MAGIC   .writeStream 
# MAGIC   .format("kafka") 
# MAGIC   .option("kafka.bootstrap.servers", "host1:port1,host2:port2") 
# MAGIC   .option("topic", "topic1") 
# MAGIC   .start()
# MAGIC )
# MAGIC ~~~

# COMMAND ----------

display(
  spark
  .readStream
  .table("{}.risk_stream_predictions".format(dbName))
)

# COMMAND ----------

# MAGIC %md Now let's gracefully terminate the streaming queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gracefully stop the streams when most data points have predictions

# COMMAND ----------

minimum_number_records = 300 # users can set this number up to 400 based on our toy test data from notebook3
while True:
  current_number_records = spark.read.table("{}.risk_stream_predictions".format(dbName)).count()
  print("Number of records with predictions", current_number_records)
  if current_number_records >= minimum_number_records:
    break
  else:
    time.sleep(10)

for s in spark.streams.active:
  print("Stopping stream")
  s.stop()


# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusions and next steps
# MAGIC In this notebook we have seen:
# MAGIC * How to load models from MLflow
# MAGIC * How to read and write streaming data from and to Delta
# MAGIC * How to divide the streaming data and make live predictions using two different models
# MAGIC 
# MAGIC In the notebook number 5 we will see how we can analyze the predictions to choose the most accurate model
# MAGIC 
# MAGIC <a href="$./5. AB testing metrics">Go to notebook 5: AB testing metrics</a>
