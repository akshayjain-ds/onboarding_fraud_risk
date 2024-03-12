# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Reject Inferencing
# MAGIC This notebook is used to score the rejected members and infer the target label along model predicted scores as weight. The idea is to use this inferred label information and train another model and learn distribution on RMs (as scoring will be don e on RMs in real time)
# MAGIC
# MAGIC Document: https://tideaccount.atlassian.net/wiki/spaces/DATA/pages/3925016577/Reject+Inference

# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

df = pd.read_csv(data_location + f"app_fraud_feature_encoded_dataset_{start_date}_{end_date}" + ".csv",
                 dtype={id1: "str", id2: "str"})
df.set_index(id1, inplace=True)
df[date_feature] = pd.to_datetime(df[date_feature]).apply(lambda x: x.date())
df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

from ast import literal_eval
df['company_sic'] = df['company_sic'].apply(lambda x: literal_eval(x))
df['applicant_nationality'] = df['applicant_nationality'].apply(lambda x: literal_eval(x))
nationality_count = df['applicant_nationality'].apply(lambda x: len(x)).max()
sic_count = df['company_sic'].apply(lambda x: len(x)).max()
nationality_count, sic_count

# COMMAND ----------

train_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(train_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(train_end_date))]
test_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(test_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(test_end_date))]
val_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(val_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(val_end_date))]

train_dataset.shape, test_dataset.shape, val_dataset.shape

# COMMAND ----------

train_dataset['data_type'] = 'train'
test_dataset['data_type'] = 'test'
val_dataset['data_type'] = 'val'
train_dataset.shape, test_dataset.shape, val_dataset.shape

# COMMAND ----------

df = pd.concat([train_dataset, test_dataset, val_dataset])
df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

df.groupby(['is_approved'])[[*get_decision_categorical_features()]].mean()

# COMMAND ----------

df.head()

# COMMAND ----------

with open(artefact_location + "cal_km_appf_model.pkl", 'rb') as f:
  cal_xgb_model = pickle.load(f)
cal_xgb_model

# COMMAND ----------

xgb_model = copy.deepcopy(cal_xgb_model.estimator)
xgb_model

# COMMAND ----------

df['appf_rating_raw_uncal'] = np.around(xgb_model.predict_proba(df[[*get_decision_features()]])[:, 1]*1000, decimals=0)
df['appf_rating_raw'] = np.around(cal_xgb_model.predict_proba(df[[*get_decision_features()]])[:, 1]*1000, decimals=0)

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(df[target_b], 
                                 df['appf_rating_raw'], sample_weight=df[km_indicator])
plot_roc_auc(fpr, tpr, 'df_appf', 'b', False)

# COMMAND ----------

app_df = df[df['is_approved']==1]
rej_df_1 = df[df['is_approved']==0]
rej_df_0 = df[df['is_approved']==0]
app_df.shape, rej_df_1.shape, rej_df_0.shape

# COMMAND ----------

app_df['weights'] = 1.0
rej_df_1['weights'] = rej_df_1['appf_rating_raw']/1000
rej_df_1['is_app_fraud'] = 1.0
rej_df_0['weights'] = 1.0 - (rej_df_0['appf_rating_raw']/1000)
rej_df_0['is_app_fraud'] = 0.0

# COMMAND ----------

rej_df_1.head()

# COMMAND ----------

rej_df_0.head()

# COMMAND ----------

concat_df = pd.concat([app_df, rej_df_0, rej_df_1])
concat_df.shape

# COMMAND ----------

concat_df.head()

# COMMAND ----------

concat_df.groupby(['is_approved', 'is_app_fraud'])['weights'].count()

# COMMAND ----------

concat_df.groupby(['is_approved', 'is_app_fraud'])['weights'].sum()

# COMMAND ----------

np.average(concat_df[target_b], weights=concat_df[km_indicator]), np.average(concat_df[target_b], weights=concat_df['weights'])

# COMMAND ----------

concat_df.drop(columns=['appf_rating_raw_uncal','appf_rating_raw'], inplace=True)
concat_df.shape

# COMMAND ----------

np.average(concat_df[target_b], weights=concat_df['weights'])

# COMMAND ----------

concat_df.columns

# COMMAND ----------

concat_df.groupby(['data_type'])[['is_approved']].count()

# COMMAND ----------

import os
# os.mkdir(data_location)
concat_df.to_csv(data_location + "app_fraud_rej_inf_feature_encoded_dataset_{start_date}_{end_date}" + ".csv")

# COMMAND ----------

