# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Target Encodings
# MAGIC This notebook is used to encode all the potential feartures that will be inputs for model training in downstream notebooks

# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./queries

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

df = pd.read_csv(data_location + f"app_fraud_feature_dataset_{start_date}_{end_date}.csv",
                 dtype={id1: "str", id2: "str"})
df.set_index(id1, inplace=True)
df[date_feature] = pd.to_datetime(df[date_feature]).apply(lambda x: x.date())
df.shape

# COMMAND ----------


from ast import literal_eval
df['company_sic'] = df['company_sic'].apply(lambda x: literal_eval(x))
df = df.merge(df[['company_sic']].apply(lambda row: {f"company_sic_{i}": j for i, j in enumerate(row['company_sic'])}, axis=1, result_type='expand'), left_index=True, right_index=True)

df['applicant_nationality'] = df['applicant_nationality'].apply(lambda x: literal_eval(x))
df = df.merge(df[['applicant_nationality']].apply(lambda row: {f"applicant_nationality_{i}": j for i, j in enumerate(row['applicant_nationality'])}, axis=1, result_type='expand'), left_index=True, right_index=True)

df.shape

# COMMAND ----------

nationality_count = df['applicant_nationality'].apply(lambda x: len(x)).max()
sic_count = df['company_sic'].apply(lambda x: len(x)).max()
nationality_count, sic_count

# COMMAND ----------

mb_df = spark_connector(multi_business_query.format(from_date=start_date, to_date=end_date))
mb_df = mb_df.toDF(*[c.lower().split('.')[-1] for c in mb_df.columns])
mb_df.count()

# COMMAND ----------

mb_df = mb_df.toPandas()
mb_df.set_index(id1, inplace=True)
mb_df.shape

# COMMAND ----------

mb_df.head()

# COMMAND ----------

df = df.merge(mb_df, left_index=True, right_index=True, how = 'left')
df['is_existing_user'] = df['is_existing_user'].fillna(0).astype(int)
df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

train_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(train_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(train_end_date))]
test_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(test_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(val_end_date))]

train_dataset.shape, test_dataset.shape

# COMMAND ----------

train_labels = train_dataset[[target_b, target_c]].apply(pd.to_numeric)
test_labels = test_dataset[[target_b, target_c]].apply(pd.to_numeric)

w_train = train_dataset[km_indicator]
w_test = test_dataset[km_indicator]

print(train_dataset.shape, train_labels.shape, w_train.shape, "\n",
      test_dataset.shape, test_labels.shape, w_test.shape) 

# COMMAND ----------

print(np.average(train_labels[target_b], weights=w_train), 
      np.average(test_labels[target_b], weights=w_test))

# COMMAND ----------

test_dataset[[*get_input_features()]].head()

# COMMAND ----------

cv = 5
leaf_size = int(train_dataset[w_train.apply(bool)].shape[0]*0.01*(cv-1)/cv)
print(leaf_size)

p = train_labels[w_train.apply(bool)][target_b].mean()
std = np.around(np.sqrt(p*(1-p)), decimals=2)
print(std)

# appf_encoder = ce.WOEEncoder(cols = [*get_input_categorical_appf_features()], sigma=std, randomized=True)
# appf_encoder = ce.TargetEncoder(cols = [*get_input_categorical_appf_features()][:-2], min_samples_leaf=leaf_size, smoothing=10)
# appf_encoder = ce.MEstimateEncoder(cols = [*get_input_categorical_appf_features()][:-2], sigma=std, randomized=True, m=10)
# appf_encoder = ce.JamesSteinEncoder(cols = [*get_input_categorical_appf_features()][:-2], sigma=std, randomized=True, model='beta')
# appf_encoder = ce.GLMMEncoder(cols = [*get_input_categorical_appf_features()][:-2], sigma=std, randomized=True, binomial_target=True)

# appf_encoder = cew.NestedCVWrapper(encoder, cv=cv, random_state=123)

# COMMAND ----------


# learn continuous target encodings based on GLMM technique for categorical features with high cardinality

def create_var(var_name, var_value):
    globals()[var_name] = var_value

glmm_encoders = {}

for input_feature in [*get_input_categorical_features()[:-2]]:
  
  create_var(f"glmm_{input_feature}", 
             ce.GLMMEncoder(cols = [input_feature], sigma=std, randomized=True, binomial_target=True))
  
  eval(f"glmm_{input_feature}").fit(train_dataset[w_train.apply(bool)][f'{input_feature}'], 
                                  train_labels[w_train.apply(bool)]['is_app_fraud'])
  
  glmm_encoders[f'{input_feature}'] = eval(f"glmm_{input_feature}")

glmm_applicant_nationality = ce.GLMMEncoder(cols = ['applicant_nationality'], 
                                     sigma=std, randomized=True, binomial_target=True, handle_missing='return_nan')
glmm_applicant_nationality.fit(train_dataset[w_train.apply(bool)]['applicant_nationality_0'].rename("applicant_nationality"), train_dataset[w_train.apply(bool)]['is_app_fraud'])

glmm_encoders[f'applicant_nationality'] = glmm_applicant_nationality

glmm_company_sic = ce.GLMMEncoder(cols = ['company_sic'],
                             sigma=std, randomized=True, binomial_target=True,
                             handle_missing='return_nan')
glmm_company_sic.fit(train_dataset[w_train.apply(bool)]['company_sic_0'].rename("company_sic"),
                train_dataset[w_train.apply(bool)][target_b])
glmm_encoders[f'company_sic'] = glmm_company_sic

glmm_encoders

# COMMAND ----------

with open(artefact_location + "app_km_glmm_encoders.pkl", 'wb') as f:
  pickle.dump(glmm_encoders, f, pickle.HIGHEST_PROTOCOL)

# COMMAND ----------


# score the learned GLMM based target encodings to create numerical features transformed from categorical features

for key, encoder in glmm_encoders.items():
  
  if key not in ['applicant_nationality', 'company_sic']:

    train_dataset[f"{key}_encoded"] = encoder.transform(train_dataset[f"{key}"])
    test_dataset[f"{key}_encoded"] = encoder.transform(test_dataset[f"{key}"])

  elif key in ['applicant_nationality']:
    
    cols = [f'{key}_0', f'{key}_1']
    for i, col in enumerate(cols):

      train_dataset[f'{col}_encoded'] = encoder.transform(train_dataset[col].rename(f"{key}"))
      test_dataset[f'{col}_encoded'] = encoder.transform(test_dataset[col].rename(f"{key}"))

    train_dataset[f'{key}_encoded'] = train_dataset[[f'{col}_encoded' for col in cols]].max(axis=1)
    test_dataset[f'{key}_encoded'] = test_dataset[[f'{col}_encoded' for col in cols]].max(axis=1)

  elif key in ['company_sic']:

    cols = ['company_sic_0', 'company_sic_1', 'company_sic_2', 'company_sic_3', 'company_sic_4']
    for i, col in enumerate(cols):

      train_dataset[f'{col}_encoded'] = encoder.transform(train_dataset[col].rename(f"{key}"))
      test_dataset[f'{col}_encoded'] = encoder.transform(test_dataset[col].rename(f"{key}"))

    train_dataset[f"{key}_encoded"] = train_dataset[[f'{col}_encoded' for col in cols]].max(axis=1)
    test_dataset[f"{key}_encoded"] = test_dataset[[f'{col}_encoded' for col in cols]].max(axis=1)

  else:

    pass

# COMMAND ----------


# Create a combined NOB feature from ICC and SIC

train_dataset['company_nob_encoded'] = train_dataset[['company_icc_encoded', 'company_sic_encoded']].max(axis=1)
test_dataset['company_nob_encoded'] = test_dataset[['company_icc_encoded', 'company_sic_encoded']].max(axis=1)

# COMMAND ----------

feature = 'company_nob_encoded'
mu, sig = weighted_avg_and_std(train_dataset[feature], w_train)
print(mu, sig)

# COMMAND ----------

# adding a controlled random noise to the feature (only in training set) that has unrealistically high information value. It will end up eating all the feature importance in the model and make it highly dependant on a single feature

train_dataset[f'{feature}'] = train_dataset[feature] + np.random.normal(
                                        0, 
                                        np.sqrt(sig)/2, 
                                        train_dataset[feature].shape)
mu, sig = weighted_avg_and_std(train_dataset[feature], w_train)
print(mu, sig)

# COMMAND ----------

train_dataset.shape, train_labels.shape, w_train.shape, test_dataset.shape, test_labels.shape, w_test.shape

# COMMAND ----------


# train the OptBinning algorithm to transform features from GLMM target encodings to WOE encodings for making features more generic and easy rank

def create_var(var_name, var_value):
    globals()[var_name] = var_value

optb_encoders = {}

for encoded_feature in [*get_decision_categorical_features()]:

  raw_feature = "_".join(encoded_feature.split("_")[:-1])
  
  create_var(f"optb_{raw_feature}", 
             OptimalBinning(name=f'{raw_feature}_encoded', dtype="numerical", solver="cp"))
  
  eval(f"optb_{raw_feature}").fit(train_dataset[w_train.apply(bool)][f'{raw_feature}_encoded'], 
         train_labels[w_train.apply(bool)][target_b])
  
  optb_encoders[f'{raw_feature}'] = eval(f"optb_{raw_feature}")

for raw_feature in ['age_at_completion', 
                    'company_age_at_completion',
                    'applicant_years_to_id_expiry']:
  
  create_var(f"optb_{raw_feature}", 
             OptimalBinning(name=f'{raw_feature}', dtype="numerical", solver="cp"))
  
  eval(f"optb_{raw_feature}").fit(train_dataset[w_train.apply(bool)][f'{raw_feature}'], 
         train_labels[w_train.apply(bool)][target_b])
  
  optb_encoders[f'{raw_feature}'] = eval(f"optb_{raw_feature}")
  
optb_encoders

# COMMAND ----------

with open(artefact_location + "app_km_optb_encoders.pkl", 'wb') as f:
  pickle.dump(optb_encoders, f, pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

with open(artefact_location + "app_km_optb_encoders.pkl", 'rb') as f:
  optb_encoders = pickle.load(f)
optb_encoders

# COMMAND ----------

pd.isnull(train_dataset[[*get_input_features()]]).sum()/train_dataset.shape[0]

# COMMAND ----------


# score the learned OptBinning features from GLMM target encoded features to to make them more generic

for key, value in optb_encoders.items():

  try:
    train_dataset[f"{key}_encoded"] = value.transform(train_dataset[f"{key}_encoded"], 
                                                      metric="woe", metric_missing='empirical')
    test_dataset[f"{key}_encoded"] = value.transform(test_dataset[f"{key}_encoded"], 
                                                     metric="woe", metric_missing='empirical')
  
  except:
    train_dataset[f"{key}_encoded"] = value.transform(train_dataset[f"{key}"], 
                                                      metric="woe", metric_missing='empirical')
    test_dataset[f"{key}_encoded"] = value.transform(test_dataset[f"{key}"], metric="woe", 
                                                     metric_missing='empirical')

# COMMAND ----------

for key, value in optb_encoders.items():
  print(value.binning_table.build())
  print(value.binning_table.plot(metric="woe", add_missing=True))

# COMMAND ----------

train_dataset['rule_applicant_idtype_brp'] = train_dataset['applicant_id_type'].apply(lambda x: x.__contains__("Residence_Permit"))*1
test_dataset['rule_applicant_idtype_brp'] = test_dataset['applicant_id_type'].apply(lambda x: x.__contains__("Residence_Permit"))*1

# COMMAND ----------

df = pd.concat([train_dataset, test_dataset])
df.shape

# COMMAND ----------

pd.isnull(df[[*get_decision_features()]]).sum()

# COMMAND ----------

import os
# os.mkdir(outdir)
df.to_csv(data_location + f"app_fraud_feature_encoded_dataset_{start_date}_{end_date}" + ".csv")

# COMMAND ----------

