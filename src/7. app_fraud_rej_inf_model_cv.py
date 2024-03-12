# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Model Time wise K fold Cross Validation
# MAGIC This notebook is for validating the model on a 5 folds based on time.
# MAGIC
# MAGIC Document: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4

# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

df = pd.read_csv(data_location + "app_fraud_rej_inf_feature_encoded_dataset_{start_date}_{end_date}" + ".csv",
                 dtype={id1: "str", id2: "str"})
df.set_index(id1, inplace=True)
df[date_feature] = pd.to_datetime(df[date_feature]).apply(lambda x: x.date())
df.shape

# COMMAND ----------

df[id1] = df.index
df.set_index(date_feature, inplace=True)
df[date_feature] = df.index
df.sort_index(inplace=True)
df.head()

# COMMAND ----------

df.head()

# COMMAND ----------

pd.isnull(df).sum()

# COMMAND ----------

from hyperopt import hp
import numpy as np
from sklearn.metrics import log_loss
random_state=111
xgb_param_grid = {
    'booster' : 'gbtree',       
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.51, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(3, 6, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
    'colsample_bynode': hp.choice('colsample_bynode', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.6, 0.9),
    'n_estimators':     hp.choice('n_estimators', np.arange(250, 300, 1, dtype=int)),
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

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit()
print(tscv)

# COMMAND ----------

with open(artefact_location + "appf_rm_thresholds.json", 'r') as f:
  thresholds_json = json.load(f)
thresholds_json

# COMMAND ----------

def create_predictions_df(dataset: pd.DataFrame, y_pred: np.ndarray, w: pd.Series) -> pd.DataFrame:

  model_decisions = pd.DataFrame(dataset[target_b])
  model_decisions[target_c] = dataset[target_c]
  model_decisions['appf_rating_raw'] = y_pred
  model_decisions['weights'] = w
  model_decisions[km_indicator] = dataset[km_indicator]
  model_decisions[f'{target_b}_w'] = model_decisions[target_b] * model_decisions['weights']

  return model_decisions

# COMMAND ----------

train_roc_auc_5fold_comp = []
test_roc_auc_5fold_comp = []
train_roc_auc_5fold_appr = []
test_roc_auc_5fold_appr = []
test_risk_ditribution_5fold = {}


for i, (train_index, test_index) in enumerate(tscv.split(df)):

  print(f"Fold {i+1}:")
  print(f"  Train: observations={len(train_index)}, between {'{:%Y-%m-%d}'.format(df.iloc[train_index].index.min())} and {'{:%Y-%m-%d}'.format(df.iloc[train_index].index.max())}")
  print(f"  Test: observations={len(test_index)}, between {'{:%Y-%m-%d}'.format(df.iloc[test_index].index.min())} and {'{:%Y-%m-%d}'.format(df.iloc[test_index].index.max())}")

  train_dataset = df.iloc[train_index]
  test_dataset = df.iloc[test_index]
  
  train_labels =  train_dataset[[target_b, target_c]]
  test_labels = test_dataset[[target_b, target_c]]
  
  w_train = train_dataset['weights']
  w_test = test_dataset['weights']

  km_train = train_dataset[km_indicator]
  km_test = test_dataset[km_indicator]

  scaler = MinMaxScaler()
  wv_train_app = w_train.copy()
  wv_train_app[~km_train.apply(bool)] = 0
  wv_train_app = pd.Series(
    scaler.fit_transform(
      w_train.values.reshape(-1,1) + scaler.fit_transform(train_labels[[target_c]])
      ).flatten(),
    index=w_train.index)
  wv_train_app = wv_train_app * (w_train.sum()/wv_train_app.sum())

  wv_train_rej = w_train.copy()
  wv_train_rej[km_train.apply(bool)] = 0  
  wv_train = wv_train_app + wv_train_rej

  xgb_para['param_grid']['scale_pos_weight'] = (wv_train.sum() - sum(train_labels[target_b] * wv_train)) / sum(train_labels[target_b] * wv_train)
  xgb_param_grid['min_child_weight'] = hp.choice('min_child_weight', np.arange(int(train_dataset['weights'].sum()*0.0025), int(train_dataset['weights'].sum()*0.0075), 50, dtype=int))
  
  obj = HPOpt(train_dataset[[*get_decision_features()] + [date_feature]], 
            train_labels[target_b], 
            test_dataset[[*get_decision_features()] + [date_feature]], 
            test_labels[target_b], 
            date_var_name = date_feature, w_train=wv_train, w_test=w_test)
  
  xgb_trials = SparkTrials()
  xgb_opt = obj.process(fn_name='xgb_model', space=xgb_para, trials=xgb_trials, algo=tpe.suggest, max_evals=100, random_state = np.random.default_rng(random_state))
  
  xgb_model = getBestModelfromTrials(xgb_trials)
  
  cal_xgb_model = CalibratedClassifierCV(xgb_model, cv ='prefit')
  cal_xgb_model.fit(train_dataset[[*get_decision_features()]],
                  train_labels[target_b], sample_weight = w_train)
  
  y_train_pred = np.around(cal_xgb_model.predict_proba(train_dataset[[*get_decision_features()]])[:, 1]*1000, 0)
  y_test_pred = np.around(cal_xgb_model.predict_proba(test_dataset[[*get_decision_features()]])[:, 1]*1000, 0)
   
  fpr, tpr, roc_thresholds_fold = metrics.roc_curve(train_labels[target_b], 
                                                    y_train_pred,
                                                    sample_weight = w_train)
  train_roc_auc_5fold_comp.append(metrics.auc(fpr, tpr))


  fpr, tpr, roc_thresholds_fold = metrics.roc_curve(test_labels[target_b], 
                                                    y_test_pred,
                                                    sample_weight = w_test)
  test_roc_auc_5fold_comp.append(metrics.auc(fpr, tpr))

  
  fpr, tpr, roc_thresholds_fold = metrics.roc_curve(train_labels[km_train.apply(bool)][target_b], 
                                                    y_train_pred[km_train.apply(bool)], sample_weight = w_train[km_train.apply(bool)])
  train_roc_auc_5fold_appr.append(metrics.auc(fpr, tpr))


  fpr, tpr, roc_thresholds_fold = metrics.roc_curve(test_labels[km_test.apply(bool)][target_b],
                                                    y_test_pred[km_test.apply(bool)],
                                                    sample_weight = w_test[km_test.apply(bool)])
  test_roc_auc_5fold_appr.append(metrics.auc(fpr, tpr))

  model_decisions_test = create_predictions_df(test_dataset, y_test_pred, w_test)
  test_thresholds = cal_thresholds(model_decisions_test, 
               0, 100, 10,
               100, 15, 45)
  
  test_risk_ditribution_5fold[f'fold_{i+1}'] = test_thresholds

# COMMAND ----------

fold_metrics = {
  "Train": {
    "RM": {
      "AUC_5fold": ['{0:.2f}'.format(auc) for auc in train_roc_auc_5fold_comp],
      "AUC_5fold_mean": np.around(np.average(train_roc_auc_5fold_comp), decimals=2), 
      "AUC_5fold_std": np.around(np.std(train_roc_auc_5fold_comp), decimals=3)
    },
    "KM": {
      "AUC_5fold": ['{0:.2f}'.format(auc) for auc in train_roc_auc_5fold_appr],
      "AUC_5fold_mean": np.around(np.average(train_roc_auc_5fold_appr), decimals=2), 
      "AUC_5fold_std": np.around(np.std(train_roc_auc_5fold_appr), decimals=3)
    }
  },
  "Test": {
    "Risk Distribution": test_risk_ditribution_5fold,
    "RM": {
      "AUC_5fold": ['{0:.2f}'.format(auc) for auc in test_roc_auc_5fold_comp],
      "AUC_5fold_mean": np.around(np.average(test_roc_auc_5fold_comp), decimals=2), 
      "AUC_5fold_std": np.around(np.std(test_roc_auc_5fold_comp), decimals=3)
    },
    "KM": {
      "AUC_5fold": ['{0:.2f}'.format(auc) for auc in test_roc_auc_5fold_appr],
      "AUC_5fold_mean": np.around(np.average(test_roc_auc_5fold_appr), decimals=2), 
      "AUC_5fold_std": np.around(np.std(test_roc_auc_5fold_appr), decimals=3)
    }
  }
}
  
fold_metrics

# COMMAND ----------

json_object = json.dumps(fold_metrics, indent = 4)
with open("/Workspace/Shared/Decisioning/Strawberries/uk_app_fraud_model/app_fraud_engine_training_v2/artefacts/5fold_cv_metrics.json", "w") as f:
    f.write(json_object)

# COMMAND ----------

print(json_object)

# COMMAND ----------

