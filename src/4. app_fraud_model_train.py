# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Train Model on KMs
# MAGIC This notebook is used train a model on KMs, as we only have target label information on KMs
# MAGIC Then this trained model will be used to infer the target on rejected RMs

# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

df = pd.read_csv(data_location + f"app_fraud_feature_encoded_dataset_{start_date}_{end_date}" + ".csv",
                 dtype={id1: "str", id2: "str"})
df.set_index(id1, inplace=True)
df[date_feature] = pd.to_datetime(df[date_feature]).apply(lambda x: x.date())
df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

pd.isnull(df).sum()

# COMMAND ----------

from ast import literal_eval
df['company_sic'] = df['company_sic'].apply(lambda x: literal_eval(x))
df['applicant_nationality'] = df['applicant_nationality'].apply(lambda x: literal_eval(x))

nationality_count = df['applicant_nationality'].apply(lambda x: len(x)).max()
sic_count = df['company_sic'].apply(lambda x: len(x)).max()
nationality_count, sic_count

# COMMAND ----------

df.head()

# COMMAND ----------

pd.isnull(df).sum()/df.shape[0]

# COMMAND ----------

train_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(train_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(train_end_date))]
test_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(test_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(test_end_date))]
val_dataset = df[(pd.to_datetime(df[date_feature]) >= pd.to_datetime(val_start_date)) & 
                   (pd.to_datetime(df[date_feature]) <= pd.to_datetime(val_end_date))]

train_dataset.shape, test_dataset.shape, val_dataset.shape

# COMMAND ----------

# test_val = (pd.to_datetime(test_start_date) <= pd.to_datetime(df[date_feature])) & (pd.to_datetime(df[date_feature]) < pd.to_datetime(val_end_date))
# test_dataset = df[test_val]
# test_dataset, val_dataset = train_test_split(test_dataset, test_size=0.5)
# train_dataset = df[~test_val]
# train_dataset.shape, test_dataset.shape, val_dataset.shape

# COMMAND ----------

train_labels = train_dataset[[target_b, target_c]].apply(pd.to_numeric)
test_labels = test_dataset[[target_b, target_c]].apply(pd.to_numeric)
val_labels = val_dataset[[target_b, target_c]].apply(pd.to_numeric)

w_train = train_dataset[km_indicator]
w_test = test_dataset[km_indicator]
w_val = val_dataset[km_indicator]

scaler = MinMaxScaler()
wv_train = pd.Series(scaler.fit_transform(w_train.values.reshape(-1,1) + scaler.fit_transform(train_labels[[target_c]])).flatten(), index=w_train.index)
wv_train = wv_train * (wv_train.shape[0]/wv_train.sum())

print(train_dataset.shape, train_labels.shape, w_train.shape, wv_train.shape, "\n",
      test_dataset.shape, test_labels.shape, w_test.shape, "\n",
      val_dataset.shape, val_labels.shape, w_val.shape) 

# COMMAND ----------

wv_train.describe()

# COMMAND ----------

train_dataset[w_train.apply(bool)].shape, test_dataset[w_test.apply(bool)].shape, val_dataset[w_val.apply(bool)].shape

# COMMAND ----------

assert train_dataset[w_train.apply(bool)].shape[0] +  test_dataset[w_test.apply(bool)].shape[0] + val_dataset[w_val.apply(bool)].shape[0] == df[df[km_indicator]==1].shape[0]
assert train_dataset.shape[0] + test_dataset.shape[0] + val_dataset.shape[0] == df.shape[0]

# COMMAND ----------

print(np.average(train_labels[target_b], weights=w_train*1),
      np.average(train_labels[target_b], weights=wv_train*1),
       
      np.average(test_labels[target_b], weights=w_test*1), 
      np.average(val_labels[target_b], weights=w_val*1))

# COMMAND ----------

from hyperopt import hp
import numpy as np
from sklearn.metrics import log_loss

scale_pos_weight = (wv_train.sum() - sum(train_labels[target_b] * wv_train)) / sum(train_labels[target_b] * wv_train)

feature_count = len([*get_decision_features()])
obs_count = train_dataset.shape[0]
random_state = 111
xgb_param_grid = {
    'booster' : 'gbtree',       
    'scale_pos_weight' : scale_pos_weight,
    'learning_rate':    hp.choice('learning_rate', np.arange(0.05, 0.51, 0.05)),
    'max_depth':        hp.choice('max_depth', np.arange(3, 5, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(int(obs_count*0.0025), int(obs_count*0.0075), 50, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
    'colsample_bynode': hp.choice('colsample_bynode', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.5, 0.8),
    'n_estimators':     hp.choice('n_estimators', np.arange(250, 500, feature_count, dtype=int)),
    'random_state': random_state
}

xgb_fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 5,
    'verbose': False
}

xgb_para = dict()
xgb_para['param_grid'] = xgb_param_grid
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred, w: log_loss(y, pred, sample_weight=w)
xgb_para

# COMMAND ----------

obj = HPOpt(train_dataset[[*get_decision_features()] + [date_feature]], 
            train_labels[target_b], 
            test_dataset[[*get_decision_features()] + [date_feature]], 
            test_labels[target_b], 
            date_var_name = date_feature, 
            w_train = wv_train*1, w_test = w_test*1)


# COMMAND ----------

xgb_trials = SparkTrials()
xgb_opt = obj.process(fn_name='xgb_model', space=xgb_para, trials=xgb_trials, algo=tpe.suggest, max_evals=100, random_state = np.random.default_rng(random_state))

# COMMAND ----------

xgb_model = getBestModelfromTrials(xgb_trials)
best_params_ = {}
all_params = dir(xgb_model)
for param in [*xgb_para['param_grid']]:
  if param in all_params:
    best_params_[param] = getattr(xgb_model, param)
  elif param not in ['scale_pos_weight', 'booster']:
    best_params_[param] = model.get_xgb_params().get(param) 
best_params_

# COMMAND ----------

cal_xgb_model = CalibratedClassifierCV(estimator=xgb_model, cv ='prefit', method='sigmoid')
cal_xgb_model.fit(train_dataset[[*get_decision_features()]],
                  train_labels[target_b], sample_weight = w_train*1)

# COMMAND ----------

y_train_pred_uncal = np.around(xgb_model.predict_proba(train_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_test_pred_uncal = np.around(xgb_model.predict_proba(test_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_val_pred_uncal = np.around(xgb_model.predict_proba(val_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)

y_train_pred = np.around(cal_xgb_model.predict_proba(train_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_test_pred = np.around(cal_xgb_model.predict_proba(test_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_val_pred = np.around(cal_xgb_model.predict_proba(val_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)

# COMMAND ----------

shap_explainer = shap.TreeExplainer(cal_xgb_model.estimator)
shap_values = shap_explainer(val_dataset[w_val.apply(bool)][[*get_decision_features()]])

# COMMAND ----------

shap.summary_plot(shap_values.values, features=val_dataset[w_val.apply(bool)][[*get_decision_features()]], feature_names=[*get_decision_features()], plot_type='dot', max_display=20, title='SHAP Feature Importance - Beeswarm Plot')

# COMMAND ----------

shap.plots.bar(shap_values, max_display=20)

# COMMAND ----------

def create_predictions_df(dataset: pd.DataFrame, y_pred: np.ndarray, w: pd.Series) -> pd.DataFrame:

  model_decisions = pd.DataFrame(dataset[target_b])
  model_decisions[target_c] = dataset[target_c]
  model_decisions['appf_rating_raw'] = y_pred
  model_decisions['weights'] = w

  return model_decisions

model_decisions_train = create_predictions_df(train_dataset, y_train_pred, w_train)
model_decisions_test = create_predictions_df(test_dataset, y_test_pred, w_test)
model_decisions_val = create_predictions_df(val_dataset, y_val_pred, w_val)

print(model_decisions_train.shape, model_decisions_train['appf_rating_raw'].mean(), "\n",
      model_decisions_test.shape, model_decisions_test['appf_rating_raw'].mean(), "\n",
      model_decisions_val.shape, model_decisions_val['appf_rating_raw'].mean(), "\n")

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(train_labels[target_b], y_train_pred, sample_weight=w_train)
train_roc_auc = plot_roc_auc(fpr, tpr, 'Train_appf', 'b', False)

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(test_labels[target_b], y_test_pred, sample_weight=w_test)
test_roc_auc = plot_roc_auc(fpr, tpr, 'Test_appf', 'b', False)
fpr, tpr, thresholds = roc_curve(val_labels[target_b], y_val_pred, sample_weight=w_val)
plot_roc_auc(fpr, tpr, 'Val_appf', 'g', False)

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(model_decisions_train[target_b], y_train_pred, sample_weight=w_train)
plot_precision_recall(precision, recall, "Train_appf", 'b', False)

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(model_decisions_test[target_b], y_test_pred, sample_weight=w_test)
plot_precision_recall(precision, recall, "Test_appf", 'b', False)

precision, recall, thresholds = precision_recall_curve(model_decisions_val[target_b], y_val_pred, sample_weight=w_val)
plot_precision_recall(precision, recall, "Val_appf", 'g', False)

# COMMAND ----------

with open(artefact_location + "cal_km_appf_model.pkl", 'wb') as f:
  pickle.dump(cal_xgb_model, f, pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

