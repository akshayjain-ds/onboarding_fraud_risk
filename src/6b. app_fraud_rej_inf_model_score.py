# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Scoring the final trained model on a batch of data
# MAGIC This notebook is used to score the trained model on a batch dataset only for any further analytical purposes
# MAGIC

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

w_train.sum(), w_test.sum(), w_val.sum()

# COMMAND ----------

print(np.average(train_labels[target_b], weights=w_train), 
      np.average(train_labels[target_b], weights=wv_train), 
      np.average(test_labels[target_b], weights=w_test), 
      np.average(val_labels[target_b], weights=w_val))

# COMMAND ----------

(w_train==1).sum(), (w_test==1).sum(), (w_val==1).sum()

# COMMAND ----------

print(np.average(train_labels[target_b], weights=np.where(w_train==1, 1, 0)), 
      np.average(test_labels[target_b], weights=np.where(w_test==1, 1, 0)), 
      np.average(val_labels[target_b], weights=np.where(w_val==1, 1, 0)))

# COMMAND ----------

with open(artefact_location + "cal_rm_appf_model.pkl", 'rb') as f:
  cal_xgb_model = pickle.load(f)
xgb_model = cal_xgb_model.estimator
cal_xgb_model

# COMMAND ----------

y_train_pred_uncal = np.around(xgb_model.predict_proba(train_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_test_pred_uncal = np.around(xgb_model.predict_proba(test_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_val_pred_uncal = np.around(xgb_model.predict_proba(val_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)

y_train_pred = np.around(cal_xgb_model.predict_proba(train_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_test_pred = np.around(cal_xgb_model.predict_proba(test_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)
y_val_pred = np.around(cal_xgb_model.predict_proba(val_dataset[[*get_decision_features()]])[:, 1]*1000, decimals=0)

# COMMAND ----------

with open(artefact_location + "appf_rm_shap_explainer.pkl", 'rb') as f:
  shap_explainer = pickle.load(f)
shap_explainer

# COMMAND ----------


shap_values = shap_explainer(val_dataset[(val_dataset[km_indicator]==1) | (val_dataset[target_b]==1)][[*get_decision_features()]])

# COMMAND ----------

fig = shap.summary_plot(shap_values.values, features=val_dataset[(val_dataset[km_indicator]==1) | (val_dataset[target_b]==1)][[*get_decision_features()]], feature_names=[*get_decision_features()], plot_type='dot', max_display=20, title='SHAP Feature Importance - Beeswarm Plot', show=False)

# COMMAND ----------

fig = shap.plots.bar(shap_values, max_display=20, show=False)

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
plt.show()
plt.close(fig)



# COMMAND ----------

fig = shap.plots.waterfall(shap_values[371], max_display=20, show=False)

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

model_decisions_oot = pd.concat([model_decisions_train, model_decisions_test, model_decisions_val])
model_decisions_oot = model_decisions_oot[(pd.to_datetime(model_decisions_oot[date_feature]) >= pd.to_datetime('2022-01-01')) & 
                                          (pd.to_datetime(model_decisions_oot[date_feature]) <= pd.to_datetime('2022-06-30'))]
model_decisions_oot[model_decisions_oot[km_indicator]==1][target_b].mean()

# COMMAND ----------

model_decisions_oot = pd.concat([model_decisions_train, model_decisions_test, model_decisions_val])
model_decisions_oot = model_decisions_oot[(pd.to_datetime(model_decisions_oot[date_feature]) >= pd.to_datetime('2022-07-01')) & 
                                          (pd.to_datetime(model_decisions_oot[date_feature]) <= pd.to_datetime('2022-12-31'))]
model_decisions_oot[model_decisions_oot[km_indicator]==1][target_b].mean()

# COMMAND ----------

model_decisions_oot = pd.concat([model_decisions_train, model_decisions_test, model_decisions_val])
model_decisions_oot = model_decisions_oot[(pd.to_datetime(model_decisions_oot[date_feature]) >= pd.to_datetime('2023-01-01')) & 
                                          (pd.to_datetime(model_decisions_oot[date_feature]) <= pd.to_datetime('2023-06-30'))]
model_decisions_oot[model_decisions_oot[km_indicator]==1][target_b].mean()

# COMMAND ----------

model_decisions_oot[model_decisions_oot[km_indicator]==1][target_b].mean()

# COMMAND ----------

roc_thresholds = [116.0, 22.0, 5.0, 3.0, 1.0]
roc_thresholds


# COMMAND ----------

thresholds = {'Medium -> High': {'threshold': roc_thresholds[1],
  'Precision': calc_prec_recall_gt_threshold(model_decisions_oot, roc_thresholds[1])[0],
  'Recall': calc_prec_recall_gt_threshold(model_decisions_oot, roc_thresholds[1])[1],
  'group_size': calc_prec_recall_gt_threshold(model_decisions_oot, roc_thresholds[1])[2]},
 'Low -> Medium': {'threshold': roc_thresholds[-2],
  'Precision': calc_prec_recall_lt_threshold(model_decisions_oot, roc_thresholds[-2])[0],
  'Recall': calc_prec_recall_lt_threshold(model_decisions_oot, roc_thresholds[-2])[1],
  'group_size': calc_prec_recall_lt_threshold(model_decisions_oot, roc_thresholds[-2])[2]}
              }
thresholds

# COMMAND ----------

model_decisions_oot[model_decisions_oot[km_indicator]==1][target_b].mean()

# COMMAND ----------

2.626/(100*model_decisions_oot[model_decisions_oot[km_indicator]==1][target_b].mean())

# COMMAND ----------

0.148/(100*model_decisions_oot[model_decisions_oot[km_indicator]==1][target_b].mean())

# COMMAND ----------

report = lift_report(model_decisions_val[target_b], model_decisions_val['appf_rating_raw'], weights=model_decisions_val[km_indicator], n=10)
report

# COMMAND ----------

report['risk_category'] = report['cum_precision_w'].apply(lambda x: 'High' if x > 0.015 else ("Low" if x < 0.005 else "Medium"))
report['risk_category'].value_counts()

# COMMAND ----------

report.groupby(['risk_category'])[['']]

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


# COMMAND ----------

print(performance_metrics)

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(train_labels[target_b], y_train_pred, sample_weight=w_train)
plot_precision_recall(precision, recall, "Train_appf")

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(test_labels[target_b], y_test_pred, sample_weight=w_test)
plot_precision_recall(precision, recall, "Test_appf", 'g', False)

precision, recall, thresholds = precision_recall_curve(val_labels[target_b], y_val_pred, sample_weight=w_val)
plot_precision_recall(precision, recall, "Val_appf", 'b', False)


# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(train_labels[is_approved_train.apply(bool)][target_b], y_train_pred[is_approved_train.apply(bool)], sample_weight=w_train[is_approved_train.apply(bool)])
plot_precision_recall(precision, recall, "Train_appf")

# COMMAND ----------

precision, recall, thresholds = precision_recall_curve(test_labels[is_approved_test.apply(bool)][target_b], y_test_pred[is_approved_test.apply(bool)])
plot_precision_recall(precision, recall, "Test_appf", 'g', False)

precision, recall, thresholds = precision_recall_curve(val_labels[is_approved_val.apply(bool)][target_b], y_val_pred[is_approved_val.apply(bool)])
plot_precision_recall(precision, recall, "Val_appf", 'b', False)


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

