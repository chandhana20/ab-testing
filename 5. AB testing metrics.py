# Databricks notebook source
# MAGIC %md
# MAGIC ## Computing metrics
# MAGIC Great, now we have a table were we store the predictions and a table where we have the ground truth of the users who received predictions (we can assume that there is such a feedback loop).
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_4.png?raw=true" width="1000"/>
# MAGIC 
# MAGIC 
# MAGIC In this notebook, we are going to compare the predictions with the actual responses for the models A and B over time. We will compute the Precision Recall AUC. We suggest users to start running this notebook a few minutes after the previous notebook 4 has started running so that you have more data points to make predictions on.
# MAGIC 
# MAGIC We will save these results in a Delta table so that we can read it from Databricks SQL. Also we will be able to track the quality of both models over time.

# COMMAND ----------

# MAGIC %run ./_resources/04_helper_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC # Import libraries

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
from scipy.stats import mannwhitneyu
from datetime import datetime
import pyspark.sql.types as T

# COMMAND ----------

# MAGIC %md
# MAGIC # Helper function

# COMMAND ----------

@pandas_udf("double", PandasUDFType.GROUPED_AGG)
def compute_metric(gt, p):
  precision, recall, thresholds = precision_recall_curve(gt, p)
  return auc(recall, precision)

# COMMAND ----------

# MAGIC %md
# MAGIC # Compute the Precision - Recall Area Under the Curve metric

# COMMAND ----------

df_pred = (
  spark
  .read
  .table("{}.risk_stream_predictions".format(dbName))
  .select("group", "id", "prediction", vector_to_array(F.col("probability")).getItem(1).alias("prob"), "timestamp")
)

df_gt = (
  spark
  .read
  .table("{}.german_credit_data".format(dbName))
  .select("id", "risk")
  .withColumn("ground_truth", F.when(F.col("risk")=="good", 0).otherwise(1))
)

df_stat_test = (
  df_gt
  .join(df_pred, on="id", how="inner")
  .cache()
  .groupby("group")
  .agg(compute_metric("ground_truth", "prediction").alias("pr_auc"))
  .na
  .drop()
)

display(df_stat_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PR AUC over time
# MAGIC Besides having a global PR AUC, we can also compute it in 1 minute buckets to see how the models perform over time

# COMMAND ----------


df_metrics = (
  df_gt
  .join(df_pred, on="id", how="inner")
  .withColumn("date_time", F.from_unixtime("timestamp", "MM-dd-yyyy HH:mm"))
  .groupby("group", "date_time")
  .agg(compute_metric("ground_truth", "prediction").alias("pr_auc"))
  .na
  .drop()
)

display(df_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Plot the PR AUC over time

# COMMAND ----------

import plotly.express as px
pd1 = df_metrics.toPandas().sort_values(by=['date_time'], ascending=[True]).reset_index(drop=True)
fig = px.line(pd1, x='date_time', y='pr_auc', line_group='group', color='group')
fig

# COMMAND ----------

# MAGIC %md
# MAGIC # Save metrics and tests to a Delta Table and visualize them with Databricks SQL

# COMMAND ----------

data = [(
  df_stat_test.toPandas().sort_values("pr_auc").head(1)["group"].values[0],
  ((spark.sql("select current_timestamp()")).collect()[0][0]),
  float(np.round(df_stat_test.toPandas().sort_values("pr_auc").head(1)["pr_auc"].values[0], 2))
  )]

df_best_model = spark.createDataFrame(data, T.StructType([
  T.StructField("best_model", T.StringType()),
  T.StructField("timestamp", T.TimestampType()),
  T.StructField("PR_AUC", T.FloatType())]
))

(
df_best_model
 .write
 .mode("append")
 .format("delta")
 .saveAsTable("{}.credit_risk_ab_testing".format(dbName))
)

# COMMAND ----------

(
  df_metrics
  .write
  .mode("append")
  .format("delta")
  .saveAsTable("{}.risk_metrics".format(dbName))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: create a dashboard on Databricks SQL 
# MAGIC 
# MAGIC With the previous tables you would be able to visualize the performance of both models over time and decide which one to promote to production:
# MAGIC 
# MAGIC Example dashboard: https://e2-demo-west.cloud.databricks.com/sql/dashboards/98b1108c-8901-484f-b82e-7675d4b2af12-risk-demo---field-demo?o=2556758628403379
# MAGIC 
# MAGIC You can also create your own by importing the dashboard from the file ./_resources/risk_demo.dbdash
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/sql_dashboard.png?raw=true" width="1300"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC In this notebook we have seen:
# MAGIC * How to compare the predictions with the ground truth and calculate the Precision Recall Area Under the Curve
# MAGIC * Create a Databaricks SQL dashboard to present the results

# COMMAND ----------

# MAGIC %md
# MAGIC # Final conclusions
# MAGIC In this demo we have seen how the Lakehouse simplifies the process to do A/B testing for credit risk. We have also explored how data engineers, data scientists and data analysts can work together on the Lakehouse to carry out end-to-end A/B testing.
