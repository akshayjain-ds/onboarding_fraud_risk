# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Data Quality Rpeort
# MAGIC This notebook will be used generate a data quality report for all full training dataset ontained from 2a and 2b

# COMMAND ----------

pip install sweetviz

# COMMAND ----------

import sweetviz as sv
import pandas as pd

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %md Select training dataset to create data quality report

# COMMAND ----------

dataset = pd.read_csv(data_location + f"app_fraud_feature_dataset_{start_date}_{end_date}.csv",
                 dtype={id1: "str", id2: "str"})
dataset.set_index(id1, inplace=True)
dataset[date_feature] = pd.to_datetime(dataset[date_feature]).apply(lambda x: x.date())
dataset.shape

# COMMAND ----------

dataset.shape

# COMMAND ----------

dataset.isna().sum()

# COMMAND ----------

dataset[target_b].value_counts()

# COMMAND ----------

dataset.head()

# COMMAND ----------

analysis = sv.analyze(dataset.drop(columns=[target_c]), target_feat = target_b)
analysis.show_html(artefact_location + 'appf_data_pofile.html')
analysis.show_html()


# COMMAND ----------

