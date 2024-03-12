# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Model with Actual (KM) with Reject Inferred Labels (RMs)
# MAGIC This notebook is used to train a model on whole dataset including KMs (Actual target) and Reject inferred target (RMs). This will be our final model
# MAGIC
# MAGIC Document: https://tideaccount.atlassian.net/wiki/spaces/DATA/pages/3925016577/Reject+Inference

# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./pre-processing

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

df = pd.read_csv(data_location + "app_fraud_rej_inf_feature_encoded_dataset_{start_date}_{end_date}" + ".csv",
                 dtype={id1: "str", id2: "str"})
df.set_index(id1, inplace=True)
df[date_feature] = pd.to_datetime(df[date_feature]).apply(lambda x: x.date())
df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

train_dataset = df[df['data_type']=='train']

# test_dataset = df[df['data_type']=='test']
# val_dataset = df[df['data_type']=='val']
test_dataset, val_dataset = train_test_split(df[df['data_type'].isin(['test', 'val'])], test_size=0.5, random_state=111)

train_labels = train_dataset[[target_b, target_c]].apply(pd.to_numeric)
test_labels = test_dataset[[target_b, target_c]].apply(pd.to_numeric)
val_labels = val_dataset[[target_b, target_c]].apply(pd.to_numeric)

w_train = train_dataset['weights']
w_test = test_dataset['weights']
w_val = val_dataset['weights']

is_approved_train = train_dataset[km_indicator]
is_approved_test = test_dataset[km_indicator]
is_approved_val = val_dataset[km_indicator]

print(train_dataset.shape, train_labels.shape, w_train.shape, "\n",
      test_dataset.shape, test_labels.shape, w_test.shape, "\n",
      val_dataset.shape, val_labels.shape, w_val.shape) 

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

wv_train_app = w_train.copy()
wv_train_app[~is_approved_train.apply(bool)] = 0

wv_train_app = pd.Series(
  scaler.fit_transform(
    w_train.values.reshape(-1,1) + scaler.fit_transform(train_labels[[target_c]])
    ).flatten(),
  index=w_train.index)
wv_train_app = wv_train_app * (w_train.sum()/wv_train_app.sum())

wv_train_rej = w_train.copy()
wv_train_rej[is_approved_train.apply(bool)] = 0  

wv_train = wv_train_app + wv_train_rej
wv_train.shape, wv_train.sum()

# COMMAND ----------

print(np.average(train_labels[target_b], weights=w_train), 
      np.average(train_labels[target_b], weights=wv_train), 
      np.average(test_labels[target_b], weights=w_test), 
      np.average(val_labels[target_b], weights=w_val))

# COMMAND ----------

corr_df = val_dataset[(val_dataset[km_indicator]==1) | (val_dataset[target_b]==1)][[*get_decision_features()]].corr('spearman')
corr_df.index = [col.replace('_encoded', '') for col in [*get_decision_features()]]
corr_df.columns = [col.replace('_encoded', '') for col in [*get_decision_features()]]
corr_df.shape

# COMMAND ----------

def _color_red_or_green(val):
    color = 'black' if val ==1 else ('red' if np.abs(val) >= 0.75 else ('orange' if np.abs(val) >= 0.5 else ('yellow' if np.abs(val) >= 0.25 else 'green')))
    return 'color: %s' % color
np.around(corr_df, decimals=3).style.applymap(_color_red_or_green).to_html(artefact_location + "feature_correlation_df.html")
np.around(corr_df, decimals=3).style.applymap(_color_red_or_green)

# COMMAND ----------

import seaborn as sb
fig = sb.heatmap(corr_df, vmin=-1, vmax=1)
plt.savefig(artefact_location + "feature_correlation_plot.png")

# COMMAND ----------

variance_inflation(val_dataset[(val_dataset[km_indicator]==1) | (val_dataset[target_b]==1)][[*get_decision_features()]])

# COMMAND ----------

from hyperopt import hp
import numpy as np
from sklearn.metrics import log_loss

scale_pos_weight = (wv_train.sum() - sum(train_labels[target_b] * wv_train)) / sum(train_labels[target_b] * wv_train)
feature_count = len([*get_decision_features()])
obs_count = train_dataset['weights'].sum()
random_state = 111

# COMMAND ----------

obj = HPOpt(train_dataset[[*get_decision_features()] + [date_feature]], 
            train_labels[target_b], 
            test_dataset[[*get_decision_features()] + [date_feature]], 
            test_labels[target_b], 
            date_var_name = date_feature, w_train=wv_train, w_test=w_test)


# COMMAND ----------

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

xgb_trials = SparkTrials()
xgb_opt = obj.process(fn_name='xgb_model', space=xgb_para, trials=xgb_trials, algo=tpe.suggest, max_evals=100, random_state = np.random.default_rng(random_state))

# COMMAND ----------

# LightGBM parameters
lgb_param_grid = {
    'scale_pos_weight' : scale_pos_weight,
    'learning_rate':    hp.choice('learning_rate', np.arange(0.05, 0.51, 0.05)),
    'max_depth':        hp.choice('max_depth', np.arange(3, 5, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(int(obs_count*0.0025), int(obs_count*0.0075), 50, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.9, 0.1)),
    'subsample':        hp.uniform('subsample', 0.6, 0.9),
    'n_estimators':     hp.choice('n_estimators', np.arange(250, 500, feature_count, dtype=int)),
    'random_state': random_state
}

lgb_fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 5,
    'verbose': False
}

lgb_para = dict()
lgb_para['param_grid'] = lgb_param_grid
lgb_para['fit_params'] = lgb_fit_params
lgb_para['loss_func' ] = lambda y, pred, w: log_loss(y, pred, sample_weight=w)
lgb_para

# COMMAND ----------

# lgb_trials = SparkTrials()
# lgb_opt = obj.process(fn_name='lgb_model', space=lgb_para, trials=lgb_trials, algo=tpe.suggest, max_evals=100, random_state = np.random.default_rng(random_state))

# COMMAND ----------

# CatBoost parameters
ctb_reg_params = {
    'scale_pos_weight' : scale_pos_weight,
    'learning_rate':     hp.choice('learning_rate', np.arange(0.05, 0.51, 0.05)),
    'max_depth':         hp.choice('max_depth', np.arange(3, 5, 1, dtype=int)),
    'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.9, 0.1)),
    'n_estimators':      hp.choice('n_estimators', np.arange(250, 500, feature_count, dtype=int)),
    'subsample':        hp.uniform('subsample', 0.6, 0.9),
    'random_state': random_state,
    'eval_metric':       'AUC',
}
ctb_fit_params = {
    'early_stopping_rounds': 5,
    'verbose': False
}
ctb_para = dict()
ctb_para['param_grid'] = ctb_reg_params
ctb_para['fit_params'] = ctb_fit_params
ctb_para['loss_func' ] = lambda y, pred, w: log_loss(y, pred, sample_weight=w)
ctb_para

# COMMAND ----------

# ctb_trials = SparkTrials()
# ctb_opt = obj.process(fn_name='ctb_model', space=ctb_para, trials=ctb_trials, algo=tpe.suggest, max_evals=100, random_state = np.random.default_rng(random_state))

# COMMAND ----------

xgb_model = getBestModelfromTrials(xgb_trials)
# lgb_model = getBestModelfromTrials(lgb_trials)
# ctb_model = getBestModelfromTrials(ctb_trials)

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
                  train_labels[target_b], sample_weight = w_train)

# COMMAND ----------

with open(artefact_location + "cal_rm_appf_model.pkl", 'wb') as f:
  pickle.dump(cal_xgb_model, f, pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

with open(artefact_location + "cal_rm_appf_model.pkl", 'rb') as f:
  cal_xgb_model = pickle.load(f)
xgb_model = cal_xgb_model.estimator
cal_xgb_model

# COMMAND ----------

model_params = json.dumps(cal_xgb_model.estimator.get_params(), default=np_encoder)
with open(artefact_location + "model_params.json", "w") as f:
    f.write(model_params)

# COMMAND ----------

y_train_pred_uncal = np.around(xgb_model.predict_proba(train_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_test_pred_uncal = np.around(xgb_model.predict_proba(test_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_val_pred_uncal = np.around(xgb_model.predict_proba(val_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)

y_train_pred = np.around(cal_xgb_model.predict_proba(train_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_test_pred = np.around(cal_xgb_model.predict_proba(test_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_val_pred = np.around(cal_xgb_model.predict_proba(val_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)

# COMMAND ----------

shap_explainer = shap.TreeExplainer(cal_xgb_model.estimator)
shap_values = shap_explainer(val_dataset[(val_dataset[km_indicator]==1) | (val_dataset[target_b]==1)][[*get_decision_features()]])

# COMMAND ----------

fig = shap.summary_plot(shap_values.values, features=val_dataset[(val_dataset[km_indicator]==1) | (val_dataset[target_b]==1)][[*get_decision_features()]], feature_names=[*get_decision_features()], plot_type='dot', max_display=20, title='SHAP Feature Importance - Beeswarm Plot', show=False)
plt.savefig(artefact_location + "feature_importance_beasworm_plot.png", bbox_inches = 'tight')

# COMMAND ----------

fig = shap.plots.bar(shap_values, max_display=20, show=False)
plt.savefig(artefact_location + "feature_importance_bar_plot.png", bbox_inches = 'tight')

# COMMAND ----------

X = val_dataset[(val_dataset[km_indicator]==1) | (val_dataset[target_b]==1)][[*get_decision_features()]]
rows = int(np.ceil(X.shape[1]/2))
cols = 2
fig, axes = plt.subplots(nrows= rows, ncols= cols, figsize=(rows*5, rows*3))
features = X.columns
row_cnt = col_cnt = 0
for feature_name in features:
  shap.dependence_plot(feature_name, shap_values.values, X, ax=axes[row_cnt][col_cnt], show=False, interaction_index=None)
  col_cnt = col_cnt + 1
  if col_cnt == 2:
    col_cnt = 0
    row_cnt = row_cnt + 1
plt.savefig(artefact_location + "feature_partial_dependence_plot.png", bbox_inches = 'tight')
plt.show()
plt.close(fig)



# COMMAND ----------

fig = shap.plots.waterfall(shap_values[371], max_display=20, show=False)
plt.savefig(artefact_location + "feature_importance_waterfall_plot.png", bbox_inches = 'tight')



# COMMAND ----------

with open(artefact_location + "appf_rm_shap_explainer.pkl", 'wb') as f:
  pickle.dump(shap_explainer, f, pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

def create_predictions_df(dataset: pd.DataFrame, date: str, y_pred: np.ndarray, y_pred_uncal: np.ndarray, w: pd.Series) -> pd.DataFrame:

  model_decisions = pd.DataFrame(dataset[target_b])
  model_decisions[date_feature] = pd.DataFrame(dataset[date_feature])
  model_decisions[target_c] = dataset[target_c]
  model_decisions['appf_rating_raw'] = y_pred
  model_decisions['appf_rating_raw_uncal'] = y_pred_uncal
  model_decisions['weights'] = w
  model_decisions[km_indicator] = dataset[km_indicator]
  model_decisions[f'{target_b}_w'] = model_decisions[target_b] * model_decisions['weights']
  model_decisions['risk_engine_decision'] = dataset['risk_category'].apply(lambda x: 1 if x == 'High' else 0)
  model_decisions['rules_engine_decision'] = dataset['rules_engine_decision'].apply(lambda x: 1 if x == 'mkyc' else 0)

  return model_decisions

model_decisions_train = create_predictions_df(train_dataset, date_feature, y_train_pred, y_train_pred_uncal, w_train)
model_decisions_test = create_predictions_df(test_dataset, date_feature, y_test_pred, y_test_pred_uncal, w_test)
model_decisions_val = create_predictions_df(val_dataset, date_feature, y_val_pred, y_val_pred_uncal, w_val)

print(model_decisions_train.shape, model_decisions_train['appf_rating_raw'].mean(), "\n",
      model_decisions_test.shape, model_decisions_test['appf_rating_raw'].mean(), "\n",
      model_decisions_val.shape, model_decisions_val['appf_rating_raw'].mean(), "\n")

# COMMAND ----------

model_decisions_test_exrapolated = pd.DataFrame(model_decisions_test.values.repeat(model_decisions_test.weights*1000, axis=0), columns=model_decisions_test.columns.tolist())
model_decisions_test_exrapolated.shape

# COMMAND ----------

from sklearn.calibration import calibration_curve
from matplotlib import pyplot
fig = pyplot.figure(1, figsize=(8, 8))
y, x = calibration_curve(model_decisions_test_exrapolated[target_b].astype(float), 
                         model_decisions_test_exrapolated['appf_rating_raw_uncal'].astype(float)/1000.0, 
                         strategy='quantile', n_bins=10)
pyplot.plot([0.0, 1.0], [0.0, 1.0], linestyle='--')
pyplot.plot(x, y, marker='.', label='UnCaliberated')
y, x = calibration_curve(model_decisions_test_exrapolated[target_b].astype(float), 
                         model_decisions_test_exrapolated['appf_rating_raw'].astype(float)/1000.0, 
                         strategy='quantile', n_bins=10)
pyplot.plot(x, y, marker='.', label='Caliberated')
pyplot.legend(loc='lower right')
pyplot.xlabel('Average probabilities')
pyplot.ylabel('Ratio of positives')
pyplot.legend(loc='upper left')

pyplot.savefig(artefact_location + "rm_calibaration_curve.png", bbox_inches = 'tight')
pyplot.show()
pyplot.close()

# COMMAND ----------

from sklearn.calibration import calibration_curve
from matplotlib import pyplot
fig = pyplot.figure(1, figsize=(8, 8))
y, x = calibration_curve(model_decisions_test[model_decisions_test[km_indicator]==1][target_b], model_decisions_test[model_decisions_test['is_approved']==1]['appf_rating_raw_uncal']/1000.0, 
                         strategy='quantile', n_bins=10)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(x, y, marker='.', label='UnCaliberated')
y, x = calibration_curve(model_decisions_test[model_decisions_test[km_indicator]==1][target_b], model_decisions_test[model_decisions_test['is_approved']==1]['appf_rating_raw']/1000.0, 
                         strategy='quantile', n_bins=10)
pyplot.plot(x, y, marker='.', label='Caliberated')
pyplot.legend(loc='lower right')
pyplot.xlabel('Average probabilities')
pyplot.ylabel('Ratio of positives')
pyplot.legend(loc='upper left')

pyplot.savefig(artefact_location + "km_calibaration_curve.png", bbox_inches = 'tight')
pyplot.show()
pyplot.close()

# COMMAND ----------

performance_metrics = {}
performance_metrics["AUC"] = {}
performance_metrics["AUC"]["RM"] = {}
performance_metrics["AUC"]["KM"] = {}
performance_metrics

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(train_labels[target_b], y_train_pred, sample_weight=w_train)
performance_metrics["AUC"]["RM"]["train"] = np.around(auc(fpr, tpr), decimals=2)
train_roc_auc = plot_roc_auc(fpr, tpr, 'Train_appf', 'b', False)

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(test_labels['is_app_fraud'], y_test_pred, sample_weight=w_test)
performance_metrics["AUC"]["RM"]["test"] = np.around(auc(fpr, tpr), decimals=2)
test_roc_auc = plot_roc_auc(fpr, tpr, 'Test_appf_rm', 'g', False)

fpr, tpr, thresholds = roc_curve(val_labels[target_b], y_val_pred, sample_weight=w_val)
performance_metrics["AUC"]["RM"]["val"] = np.around(auc(fpr, tpr), decimals=2)
plot_roc_auc(fpr, tpr, 'Val_appf_rm', 'b', False)

plt.savefig(artefact_location + "rm_val_roc_auc.png", bbox_inches = 'tight')

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(train_labels[is_approved_train.apply(bool)][target_b], y_train_pred[is_approved_train.apply(bool)], sample_weight=w_train[is_approved_train.apply(bool)])
performance_metrics["AUC"]["KM"]["train"] = np.around(auc(fpr, tpr), decimals=2)

train_roc_auc = plot_roc_auc(fpr, tpr, 'Train_appf_km', 'b', False)

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(test_labels[is_approved_test.apply(bool)][target_b], y_test_pred[is_approved_test.apply(bool)])
performance_metrics["AUC"]["KM"]["test"] = np.around(auc(fpr, tpr), decimals=2)
test_roc_auc = plot_roc_auc(fpr, tpr, 'Test_appf_km', 'g', False)

fpr, tpr, thresholds = roc_curve(val_labels[is_approved_val.apply(bool)][target_b], y_val_pred[is_approved_val.apply(bool)])
performance_metrics["AUC"]["KM"]["val"] = np.around(auc(fpr, tpr), decimals=2)
plot_roc_auc(fpr, tpr, 'Val_appf_km', 'b', False)

plt.savefig(artefact_location + "km_val_roc_auc.png", bbox_inches = 'tight')

# COMMAND ----------

print(performance_metrics)

# COMMAND ----------

performance_metrics = json.dumps(performance_metrics, indent = 4)
with open(artefact_location + "performance_metrics.json", 'w') as f:
  f.write(performance_metrics)

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(train_labels[target_b], y_train_pred, sample_weight=w_train)
plot_precision_recall(precision, recall, "Train_appf")

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(test_labels[target_b], y_test_pred, sample_weight=w_test)
plot_precision_recall(precision, recall, "Test_appf", 'g', False)

precision, recall, thresholds = precision_recall_curve(val_labels[target_b], y_val_pred, sample_weight=w_val)
plot_precision_recall(precision, recall, "Val_appf", 'b', False)

plt.savefig(artefact_location + "rm_precision_recall_curve.png", bbox_inches = 'tight')

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(train_labels[is_approved_train.apply(bool)][target_b], y_train_pred[is_approved_train.apply(bool)], sample_weight=w_train[is_approved_train.apply(bool)])
plot_precision_recall(precision, recall, "Train_appf")

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(test_labels[is_approved_test.apply(bool)][target_b], y_test_pred[is_approved_test.apply(bool)])
plot_precision_recall(precision, recall, "Test_appf", 'g', False)

precision, recall, thresholds = precision_recall_curve(val_labels[is_approved_val.apply(bool)][target_b], y_val_pred[is_approved_val.apply(bool)])
plot_precision_recall(precision, recall, "Val_appf", 'b', False)

plt.savefig(artefact_location + "km_precision_recall_curve.png", bbox_inches = 'tight')


# COMMAND ----------

roc_thresholds = [116.0, 12.0, 5.0, 3.0, 1.0]
roc_thresholds

# COMMAND ----------

thresholds = {'Medium -> High': {'threshold': roc_thresholds[1],
  'Precision': calc_prec_recall_gt_threshold(model_decisions_test, roc_thresholds[1])[0],
  'Recall': calc_prec_recall_gt_threshold(model_decisions_test, roc_thresholds[1])[1],
  'group_size': calc_prec_recall_gt_threshold(model_decisions_test, roc_thresholds[1])[2]},
 'Low -> Medium': {'threshold': roc_thresholds[-2],
  'Precision': calc_prec_recall_lt_threshold(model_decisions_test, roc_thresholds[-2])[0],
  'Recall': calc_prec_recall_lt_threshold(model_decisions_test, roc_thresholds[-2])[1],
  'group_size': calc_prec_recall_lt_threshold(model_decisions_test, roc_thresholds[-2])[2]}
              }
thresholds

# COMMAND ----------

thresholds_json = json.dumps(thresholds, indent = 4, default=np_encoder)
with open(artefact_location + "appf_rm_thresholds.json", 'w') as f:
  f.write(thresholds_json)

# COMMAND ----------

with open(artefact_location + "appf_rm_thresholds.json", 'r') as f:
  thresholds_json = json.load(f)
thresholds_json

# COMMAND ----------

medium_high_threshold, low_medium_threshold = thresholds.get("Medium -> High").get("threshold"), thresholds.get("Low -> Medium").get("threshold")
print(medium_high_threshold, low_medium_threshold)


# COMMAND ----------

model_decisions_train['appf_decision'] = model_decisions_train['appf_rating_raw'].apply(lambda x: 'High' if x >= medium_high_threshold else ('Medium' if x >= low_medium_threshold else 'Low'))
model_decisions_test['appf_decision'] = model_decisions_test['appf_rating_raw'].apply(lambda x: 'High' if x >= medium_high_threshold else ('Medium' if x >= low_medium_threshold else 'Low'))
model_decisions_val['appf_decision'] = model_decisions_val['appf_rating_raw'].apply(lambda x: 'High' if x >= medium_high_threshold else ('Medium' if x >= low_medium_threshold else 'Low'))

# COMMAND ----------

precision_recall_group_size = {}
precision_recall_group_size['RM'] = {}
precision_recall_group_size['RM']['precision'] = {}
precision_recall_group_size['RM']['recall'] = {}
precision_recall_group_size['RM']['group_size'] = {}
precision_recall_group_size['RM']['lift'] = {}
precision_recall_group_size['KM'] = {}
precision_recall_group_size['KM']['precision'] = {}
precision_recall_group_size['KM']['recall'] = {}
precision_recall_group_size['KM']['group_size'] = {}
precision_recall_group_size['KM']['lift'] = {}
precision_recall_group_size['KM']['value'] = {}
precision_recall_group_size['KM']['value']['mean'] = {}
precision_recall_group_size['KM']['value']['capture'] = {}

# COMMAND ----------

# precision rm
val_precision = model_decisions_val.groupby(['appf_decision'])['is_app_fraud_w'].sum()/model_decisions_val.groupby(['appf_decision'])['weights'].sum()
precision_recall_group_size['RM']['precision'] = json.loads(val_precision.to_json())
precision_recall_group_size['RM']['precision']

# COMMAND ----------

# precision km
val_precision = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])[target_b].mean()
json.loads(val_precision.to_json())
precision_recall_group_size['KM']['precision'] = json.loads(val_precision.to_json())
precision_recall_group_size['KM']['precision']

# COMMAND ----------

# recall rm
val_recall = model_decisions_val.groupby(['appf_decision'])[f'{target_b}_w'].sum()/model_decisions_val[f'{target_b}_w'].sum()
json.loads(val_recall.to_json())
precision_recall_group_size['RM']['recall'] = json.loads(val_recall.to_json())
precision_recall_group_size['RM']['recall']

# COMMAND ----------

# recall km
val_recall = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])[target_b].sum()/val_dataset[is_approved_val.apply(bool)][target_b].sum()
precision_recall_group_size['KM']['recall'] = json.loads(val_recall.to_json())
precision_recall_group_size['KM']['recall']

# COMMAND ----------

# lift rm
val_lift = (model_decisions_val.groupby(['appf_decision'])[f'{target_b}_w'].sum()/model_decisions_val.groupby(['appf_decision'])['weights'].sum())/(model_decisions_val[f'{target_b}_w'].sum()/model_decisions_val['weights'].sum())
precision_recall_group_size['RM']['lift'] = json.loads(val_lift.to_json())
precision_recall_group_size['RM']['lift']

# COMMAND ----------

# lift km
val_lift = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])[target_b].mean()/model_decisions_val[is_approved_val.apply(bool)][target_b].mean()
precision_recall_group_size['KM']['lift'] = json.loads(val_lift.to_json())
precision_recall_group_size['KM']['lift']

# COMMAND ----------

# group_size rm
val_group_size = model_decisions_val.groupby(['appf_decision'])['weights'].sum()/val_dataset['weights'].sum()
json.loads(val_group_size.to_json())
precision_recall_group_size['RM']['group_size'] = json.loads(val_group_size.to_json())
precision_recall_group_size['RM']['group_size']

# COMMAND ----------

# group_size km
val_group_size = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])['weights'].sum()/val_dataset[is_approved_val.apply(bool)]['weights'].sum()
json.loads(val_group_size.to_json())
precision_recall_group_size['KM']['group_size'] = json.loads(val_group_size.to_json())
precision_recall_group_size['KM']['group_size']

# COMMAND ----------

# money_impact_precision
val_money_impact_precision = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])[target_c].mean()
json.loads(val_money_impact_precision.to_json())
precision_recall_group_size['KM']['value']['mean'] = json.loads(val_money_impact_precision.to_json())
precision_recall_group_size['KM']['value']['mean']

# COMMAND ----------

# money_impact_recall
val_money_impact_recall = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])[target_c].sum()/model_decisions_val[is_approved_val.apply(bool)][target_c].sum()
json.loads(val_money_impact_recall.to_json())
precision_recall_group_size['KM']['value']['capture'] = json.loads(val_money_impact_recall.to_json())
precision_recall_group_size['KM']['value']['capture']

# COMMAND ----------

precision_recall_group_size

# COMMAND ----------

precision_recall_group_size = json.dumps(precision_recall_group_size, indent = 4)
with open(artefact_location + "precision_recall_group_size.json", 'w') as f:
  f.write(precision_recall_group_size)

# COMMAND ----------

# money_impact_val
money_impact = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])[target_c].sum()
money_impact = json.loads(money_impact.to_json())
val_estimate = money_impact.get('High')
val_estimate, 0.5*val_estimate

# COMMAND ----------

# money_impact_yearly_estimate
money_impact_yearly_estimate = model_decisions_val[is_approved_val.apply(bool)].groupby(['appf_decision'])[target_c].sum()
money_impact_yearly_estimate = json.loads(money_impact_yearly_estimate.to_json())
yearly_estimate = money_impact_yearly_estimate.get('High')
(yearly_estimate/3)*12, 0.5*(yearly_estimate/3)*12

# COMMAND ----------

