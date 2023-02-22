# Databricks notebook source
# MAGIC %md
# MAGIC # Live Stream
# MAGIC <img src="https://tdwi.org/articles/2017/08/07/-/media/TDWI/TDWI/BITW/datapipeline.jpg" width="700"/>
# MAGIC 
# MAGIC 
# MAGIC Now that we have two models trained, we will start generating streaming data to create new predictions. Run the following cell and leave the notebook running, then move to the next notebook. It will start inserting new rows into the table risk_stream_source. 
# MAGIC 
# MAGIC The source that we will use is the unused rows of the dataset used for model training. Hence we could assume that this is completely new data

# COMMAND ----------

# MAGIC %run ./_resources/02_start_streaming_sources

# COMMAND ----------

# MAGIC %md
# MAGIC <a href="$./4. Real time inference">Go to notebook 4: Real time inference</a>
