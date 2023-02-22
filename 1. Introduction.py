# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC A/B testing is a statistical method used to compare two versions of a product to determine which performs better. It is often used in the context of machine learning (ML) to compare the performance of different ML models and determine which one is the most effective at solving a specific problem.
# MAGIC 
# MAGIC A/B testing is important for machine learning because it allows organizations to compare the performance of different algorithms, feature sets, and model architectures, and identify which one is the most accurate, efficient, or effective in solving their specific problem. It can also help organizations better understand the trade-offs associated with different ML models and make informed decisions about which model is the best fit for their needs.
# MAGIC 
# MAGIC A/B testing is also important in the context of credit risk, as it allows organizations to compare the performance of different credit risk models and determine which one is the most effective at predicting the likelihood that a borrower will default on a loan. By comparing the results of different credit risk models, organizations can identify which model is the most accurate, efficient, or effective in predicting credit risk, and make informed decisions about which model is the best fit for their needs. A/B testing can also help organizations validate their credit risk models and ensure that they are robust and reliable.
# MAGIC 
# MAGIC <img src="https://ml-ops.org/img/mlops-loop-en.jpg" width="400"/>
# MAGIC 
# MAGIC In this series of notebooks, we will demostrate the following:
# MAGIC - How to do online inference in real time using Structured Streaming
# MAGIC - How to do A/B testing with two machine learning models registered using MLflow
# MAGIC - Visualize the results on a Databricks SQL dashboard
# MAGIC 
# MAGIC We will use a toy dataset related to credit risk. See the next cell for more details.
# MAGIC 
# MAGIC 
# MAGIC To acomplish this, we will setup a system that we will setup is the following:
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/sergioballesterossolanas/databricks-ab-testing/master/img/arch_1_new.png" width="1000"/>
# MAGIC 
# MAGIC With this system we will:
# MAGIC - Take credit risk data and trains two machine learning models with it. The models will predict the risk of giving the credit requested by a borrower.
# MAGIC - Register the models in MLflow model registry.
# MAGIC - Create a live stream of new credit requests. We will use a Delta table, although this system would be compatible with other technologies such as Kafka. These requests will come from the credit risk dataset for demostration purposes.
# MAGIC - Load the two trained ML models, and we will make real time predictions on new credit requests. The predictions will be saved as a Delta table (also streaming).
# MAGIC - Evaluate and compare the historical performance of both models and present the results on a Databricks SQL Dashboard
# MAGIC 
# MAGIC <a href="$./2. Model training">Go to notebook 2: Model training</a>
