# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Target Encodings
# MAGIC This notebook is used used do encode all the potential feartures that will be used for model training in downstream notebooks

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

with open(artefact_location + "app_km_glmm_encoders.pkl", 'rb') as f:
  glmm_encoders = pickle.load(f)
glmm_encoders

# COMMAND ----------

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

train_dataset['company_nob_encoded'] = train_dataset[['company_icc_encoded', 'company_sic_encoded']].max(axis=1)
test_dataset['company_nob_encoded'] = test_dataset[['company_icc_encoded', 'company_sic_encoded']].max(axis=1)

# COMMAND ----------

feature = 'company_nob_encoded'
mu, sig = weighted_avg_and_std(train_dataset[feature], w_train)
print(mu, sig)

# COMMAND ----------

train_dataset[f'{feature}'] = train_dataset[feature] + np.random.normal(
                                        0, 
                                        np.sqrt(sig)/2, 
                                        train_dataset[feature].shape)
mu, sig = weighted_avg_and_std(train_dataset[feature], w_train)
print(mu, sig)

# COMMAND ----------

train_dataset.shape, train_labels.shape, w_train.shape, test_dataset.shape, test_labels.shape, w_test.shape

# COMMAND ----------

for feature in [col for col in test_dataset.columns if col.__contains__('_risk')]:
  print(get_iv_class(test_labels[w_test.apply(bool)][target_b], test_dataset[w_test.apply(bool)][feature], feature))

# COMMAND ----------

print(get_iv_class(test_labels[w_test.apply(bool)][target_b], test_dataset[w_test.apply(bool)]['applicant_email_numeric'], 'applicant_email_numeric'))

# COMMAND ----------

print(get_iv_class(test_labels[w_test.apply(bool)][target_b], test_dataset[w_test.apply(bool)]['applicant_name_mismatch_ind'], 'applicant_name_mismatch_ind'))

# COMMAND ----------

test_dataset.groupby(['applicant_email_numeric', 'applicant_email_domain'])[[target_b]].mean()/test_dataset[target_b].mean()

# COMMAND ----------

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

test_dataset[w_test.apply(bool)].groupby(['applicant_idcountry_issue_encoded']).agg({'applicant_idcountry_issue': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['applicant_nationality_encoded']).agg({'applicant_nationality_0': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['applicant_id_type_encoded']).agg({'applicant_id_type': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['applicant_postcode_encoded']).agg({'applicant_postcode': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['company_postcode_encoded']).agg({'company_postcode': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['company_status_encoded']).agg({'company_status': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['applicant_email_domain_encoded']).agg({'applicant_email_domain': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['company_structurelevelwise_encoded']).agg({'company_structurelevelwise': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['company_nob_encoded']).agg({'company_icc': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------


test_dataset[w_test.apply(bool)].groupby(['company_nob_encoded']).agg({'company_sic_0': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['applicant_device_type_encoded']).agg({'applicant_device_type': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['is_restricted_keyword_present']).agg({'is_restricted_keyword_present': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['individual_identity_address']).agg({'individual_identity_address': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

test_dataset[w_test.apply(bool)].groupby(['individual_id_scan']).agg({'individual_id_scan': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['individual_blocklist']).agg({'individual_blocklist': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['fraud_fail']).agg({'fraud_fail': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['individual_sanctions_pep']).agg({'individual_sanctions_pep': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['business_internal_checks']).agg({'business_internal_checks': np.unique, target_b: np.mean, target_c: np.mean})

# COMMAND ----------

train_dataset[(w_train.apply(bool)) & (train_dataset['rule_company_bank']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_industry_animal_breeder']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()

# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['rule_industry_animal_breeder']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['rule_industry_animal_breeder']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['rule_industry_animal_breeder']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_applicant_singlename']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()

# COMMAND ----------

train_dataset[(w_train.apply(bool)) & (train_dataset['rule_applicant_singlename']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_idcountry_belarus']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()

# COMMAND ----------

train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_belarus']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_idcountry_portugal']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()

# COMMAND ----------

train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_portugal']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_idcountry_portugal']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()

# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_portugal']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_portugal']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_portugal']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_idcountry_russia']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()

# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_russia']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_russia']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_russia']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_idcountry_ukraine']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_ukraine']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_ukraine']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_ukraine']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['rule_idcountry_romania']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_romania']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_romania']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['rule_idcountry_romania']==1)].shape[0]

# COMMAND ----------

train_dataset['rule_applicant_idtype_brp'] = train_dataset['applicant_id_type'].apply(lambda x: x.__contains__("Residence_Permit"))*1
test_dataset['rule_applicant_idtype_brp'] = test_dataset['applicant_id_type'].apply(lambda x: x.__contains__("Residence_Permit"))*1

# COMMAND ----------


train_dataset[w_train.apply(bool)].groupby(['rule_applicant_idtype_brp']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['rule_applicant_idtype_brp']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['rule_applicant_idtype_brp']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['rule_applicant_idtype_brp']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['individual_id_scan']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['individual_id_scan']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['individual_id_scan']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['individual_id_scan']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['individual_identity_address']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['individual_identity_address']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['individual_identity_address']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['individual_identity_address']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['individual_sanctions_pep']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['individual_sanctions_pep']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['individual_sanctions_pep']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['individual_sanctions_pep']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['individual_blocklist']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['individual_blocklist']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['individual_blocklist']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['individual_blocklist']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['business_internal_checks']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['business_internal_checks']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['business_internal_checks']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['business_internal_checks']==1)].shape[0]

# COMMAND ----------

prohibited_nationalities = ["AF", "AFG", "GN" "GIN", "ML", "MLI", 
                            "SS", "SSD", "GW", "GNB", "BY", "BLR", 
                            "MM", "MMR", "SD", "SDN", "BA", "BIH", 
                            "IR", "IRN", "RU", "RUS", "UA", "UKR",
                            "KP", "PRK", "SY", "SYR", "BI", "BDI",
                            "IQ", "IRQ", "NI", "NIC", "VE", "VEN",
                            "CF", "CAF", "LB", "LBN", "YE", "YEM",
                            "CD", "COD", "LY", "LBY", "SO", "SOM", 
                            "ZW", "ZWE"]

train_dataset['prohibited_nationality'] = train_dataset['applicant_nationality'].apply(lambda x: any([True for c in x if c in prohibited_nationalities]))*1

test_dataset['prohibited_nationality'] = test_dataset['applicant_nationality'].apply(lambda x: any([True for c in x if c in prohibited_nationalities]))*1

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['prohibited_nationality']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['prohibited_nationality']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['prohibited_nationality']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['prohibited_nationality']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['is_existing_user']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['is_existing_user']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['is_existing_user']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['is_existing_user']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['applicant_name_mismatch_ind']).agg({
  'is_app_fraud': np.sum, 
  'app_fraud_amount': np.sum}
  )/train_dataset[w_train.apply(bool)][['is_app_fraud', 'app_fraud_amount']].sum()


# COMMAND ----------

(train_dataset[(w_train.apply(bool)) & (train_dataset['applicant_name_mismatch_ind']==1)].shape[0] - train_dataset[(w_train.apply(bool)) & (train_dataset['applicant_name_mismatch_ind']==1) & (train_dataset['is_app_fraud']==1)].shape[0])/train_dataset[(w_train.apply(bool)) & (train_dataset['applicant_name_mismatch_ind']==1)].shape[0]

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['company_age_at_completion_encoded']).agg({'company_age_at_completion': np.mean, 'is_app_fraud': np.mean, 'app_fraud_amount': np.mean})

# COMMAND ----------

train_dataset[w_train.apply(bool)].groupby(['applicant_years_to_id_expiry_encoded']).agg({'applicant_years_to_id_expiry': np.mean, 'is_app_fraud': np.mean, 'app_fraud_amount': np.mean})

# COMMAND ----------

value.transform([None, np.NaN], metric="woe", metric_missing='empirical')

# COMMAND ----------

corr_df = test_dataset[w_test.apply(bool)][[*get_decision_features()]].corr('spearman')
corr_df.shape

# COMMAND ----------

import seaborn as sb
dataplot = sb.heatmap(corr_df, vmin=-1, vmax=1)

# COMMAND ----------

variance_inflation(test_dataset[w_test.apply(bool)][[*get_decision_features()]])

# COMMAND ----------

train_dataset.shape, test_dataset.shape

# COMMAND ----------

assert train_dataset.shape[1] == test_dataset.shape[1]

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

