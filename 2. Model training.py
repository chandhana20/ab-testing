# Databricks notebook source
# MAGIC %md
# MAGIC # Download the dataset
# MAGIC 
# MAGIC <img src="https://thumbs.dreamstime.com/b/credit-risk-message-bubble-word-cloud-collage-business-concept-background-credit-risk-message-bubble-word-cloud-collage-business-216251701.jpg" width="600"/>
# MAGIC 
# MAGIC We will start by downloading a dataset. Our toy dataset will be the German Credit Risk dataset:
# MAGIC 
# MAGIC https://archive-beta.ics.uci.edu/dataset/144/statlog+german+credit+data
# MAGIC 
# MAGIC Credit goes to the original author Hans Hofmann.
# MAGIC 
# MAGIC A preview of the data is available here https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/german_credit_data.csv
# MAGIC 
# MAGIC We will run two notebooks that capture all the logic and will create the table "german_credit_data" with the data.

# COMMAND ----------

# MAGIC %run ./_resources/notebook_setup $reset_all_data=True

# COMMAND ----------

# MAGIC %run ./_resources/01_download_data

# COMMAND ----------

# MAGIC %md
# MAGIC # ML model training
# MAGIC The goal of this section is to load the German Credit Data dataset and train two machine learning models with it. 
# MAGIC 
# MAGIC The machine learning model will be able to predict the risk of providing loans to different people.
# MAGIC 
# MAGIC This is the typical flow that a Data Scientist would follow.
# MAGIC 
# MAGIC In the high level diagram, we will work in the highlighted section:
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_2.png?raw=true" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

import mlflow
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from pyspark.ml.functions import vector_to_array
from pandas_profiling import ProfileReport
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the German Credit Data dataset
# MAGIC This dataset contains 1000 rows and 11 columns. Each row represents one person requesting a credit. We are going to use only the first 600 rows for model training. The remaining 400 rows will be used for A/B testing (we will assume these rows come at a later point in time). Regarding the columns:
# MAGIC - The *risk* column shows if it is risky (bad) or if it is not risky (good) to provide the credit. In the real world, this column could be replaced with information if the borrower fully repaid the loan or not for example.
# MAGIC - The *id* column represents the unique id of the request
# MAGIC - The rest of the columns are properties of the person requesting the credit, as well as information about the requested credit itself (amount, duration, ...)

# COMMAND ----------

print("schema.table:", permanent_table_name)
df = spark.read.table(permanent_table_name).where(F.col("id") < 600) # Load only the first 600 rows
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Exploratory Analysis
# MAGIC To explore the dataset we can leverage the Data Profiler. To do so, you can use the Data Profiler as shown below:

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/data_profiler_gif.gif?raw=true" width="100%"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling, evaluation and tracking with MLflow
# MAGIC Great! Now that we have generated insights about the dataset and it is already structured we can proceed with the modelling.
# MAGIC 
# MAGIC Modelling is a very iterative process that might need a lot of experimentation. As shown in the diagram below. It requires preparing the data, extracting features, training a model with some specific hyperparameters and evaluate it. If the evaluation shows that the model does not meet the desired level of quality, a new iteration should be done, which might imply processing the data again, extract different or more features, training the same model with different hyperparameters or a different model, and carry out the evaluation once again.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://cdn-images-1.medium.com/max/1600/1*WjXHRFcFT--7jPRWJ9Q5Ww.jpeg" width="1000"/>
# MAGIC 
# MAGIC This process can be complex to track, and also can happen over different days, weeks or even months. For that reason MLflow is a handy framework to track the experiments:
# MAGIC 
# MAGIC https://www.mlflow.org/docs/latest/tracking.html
# MAGIC 
# MAGIC In this notebook we will use MLflow to track the training of this model:
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocess the data
# MAGIC We will split the dataset into training and test. Also we will vectorize the categorical features using a pipeline

# COMMAND ----------

df_train, df_test = df.randomSplit(weights=[0.8, 0.2], seed=42)

string_cols = ["saving_accounts", "checking_account", "purpose"]
strings_cols_index = [i + " encoded" for i in string_cols]
strings_cols_encoded = [i + " encoded" for i in strings_cols_index]

str_indexer_label = StringIndexer(
  inputCol="risk",
  outputCol="label",
  handleInvalid="skip"
)
str_indexer = StringIndexer(
  inputCols=string_cols,
  outputCols=strings_cols_index,
  handleInvalid="keep"
)
hot_encoder = OneHotEncoder(
  inputCols=strings_cols_index,
  outputCols=strings_cols_encoded,
  handleInvalid="keep"
)
vector_assembler = VectorAssembler(
  inputCols=["age", "job", "duration", "credit_amount"] + strings_cols_encoded,
  outputCol="features",
  handleInvalid="keep"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Helper function to train and evaluate the models

# COMMAND ----------

def evaluate_model(model, df_test, image_name):
  """
  Evaluates the credit risk model on a test dataset 
  and calculate the Precision Recall Area Under the Curve (PR AUC)
  """
  # Evaluate the clasifier
  evaluation = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
  )
  
  df_pred = model.transform(df_test)
  pr_auc = evaluation.evaluate(df_pred)
  print("Test set PR AUC:", pr_auc)
  
  df_probs = (
    df_pred
    .select("label", "prediction", vector_to_array(F.col("probability")).getItem(1).alias("prob"))
    .toPandas()
  )

  # precision recall curve
  precision, recall, _ = precision_recall_curve(df_probs["label"], df_probs["prob"])
  f = plt.figure(figsize=(10,7))
  plt.plot(recall, precision, lw=2)
  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.legend(loc="best")
  plt.title("precision vs. recall curve")
  plt.savefig(image_name)
  plt.show()
  
  return pr_auc

# COMMAND ----------

def train_model(ml_model, image_path, run_name, group):
  """
  Trains a ML model and registers it using MLflow
  """
  # Enable automatic logging of input samples, metrics, parameters, and models
  mlflow.spark.autolog(silent=False)
  with mlflow.start_run(run_name=run_name) as mlflow_run:
    mlflow.set_tag("group", group)
    pipeline = Pipeline(stages=[str_indexer, hot_encoder, vector_assembler, str_indexer_label, ml_model])

    # Fit the pipeline to training documents.
    model = pipeline.fit(df_train)
    pr_auc = evaluate_model(model=model, df_test=df_test, image_name=image_path)
    
    # Train on full dataset
    model = pipeline.fit(df_train.union(df_test))
    
    # Log on MLflow
    mlflow.spark.log_model(model, artifact_path="model")
    mlflow.log_metric(key="PR_AUC", value=pr_auc)
    mlflow.log_param(key="Stages", value=str(pipeline.getStages()))
    mlflow.log_artifact(image_path)
    run_id = mlflow_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model A
# MAGIC This model will use the preprocessed dataframe, and use logistic regression to classify the credit requests.

# COMMAND ----------

train_model(
  ml_model=LogisticRegression(maxIter=1000),
  image_path="_resources/pr-curve-model-a.png",
  run_name="logistic_regression",
  group="A"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model B
# MAGIC This model is similar to the previous one, but will use gradient boosted trees for the modelling. The data preprocessing remains the same.

# COMMAND ----------

train_model(
  ml_model=GBTClassifier(),
  image_path="_resources/pr-curve-model-b.png",
  run_name="gradient_boosted_trees",
  group="B"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore the experiment and registered models
# MAGIC We can explore the MLflow UI as shown in the animation below to see the information of the trained models. We will see among other things:
# MAGIC * The PR AUC score
# MAGIC * Model version
# MAGIC * Tags
# MAGIC * Stages

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/registered_models.gif?raw=true" width="100%"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # Recap and next steps
# MAGIC We have seen how simple it was to:
# MAGIC * Read a table
# MAGIC * Use the data profiler to explore the dataset
# MAGIC * Train two machine learning models with it
# MAGIC * Register the models using MLflow model registry
# MAGIC 
# MAGIC In the next notebook, we will start a streaming source using Delta Lake. We will use it for live inference.
# MAGIC 
# MAGIC <a href="$./3. Start streaming sources">Go to notebook 3: Start streaming sources</a>
