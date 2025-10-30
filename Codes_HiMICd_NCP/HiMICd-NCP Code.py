# -*- coding: utf-8 -*-

# In[Import python libs]
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn import metrics
import os
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# %% [Step 1: Input data]
# In[Load data]
###------ Load Training and Testing Data ------
Train_path = r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Trainsample.csv"
Train = pd.read_csv(Train_path)
Train_cleaned = Train.drop(columns=Train.filter(regex="Unname").columns)
Train_cleaned = Train_cleaned[~Train_cleaned.isin([-9999]).any(axis=1)]
Train = Train_cleaned
print(Train)
del  Train_path

# In[Chose data ]
# Note:Modeling each index year by year
###------ Choose the Index waiting for training ------
year = 2003  # from 2003 to 2020
Y_col = "AVP" # indices including ['AVP'9, 'DPT'10, 'MR'11, 'RH12', 'SH13', 'VPD14']

Train_Ayear = Train.loc[Train['year'].isin([year])]
data = Train_Ayear.iloc[:, 1:8]
if Y_col = "AVP":
    data['avp'] = Train_Ayear.iloc[:, 8]
if Y_col = "DPT":
    data['dpt'] = Train_Ayear.iloc[:, 9]
if Y_col = "MR":
    data['mr'] = Train_Ayear.iloc[:, 10]
if Y_col = "RH":
    data['rh'] = Train_Ayear.iloc[:, 11]
if Y_col = "SH":
    data['sh'] = Train_Ayear.iloc[:, 12]
if Y_col = "VPD":
    data['vpd'] = Train_Ayear.iloc[:, 13]
data[Y_col] = Train_Ayear[Y_col]
x_train = data.iloc[:, 0:8]
y_train = data[Y_col]

# %% [Step 2: Hyperparameter Optimization]
# Note: The value and its testing range in this parts can be set by yourself
#---------- setp 1:  n_estimators ----------
start_time1 = time.time()
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    }

data_train = lgb.Dataset(x_train, y_train)

cv_results = lgb.cv(
    params,
    data_train,
    num_boost_round=30000,
    nfold=5,
    stratified=False,
    shuffle=True,
    metrics='rmse',
    seed=0,
    callbacks=[
        lgb.early_stopping(500),  # 早停
        lgb.log_evaluation(period=100)  # 替代 verbose_eval，每 100 轮打印一次日志
    ]
)
best_estimator = len(cv_results['valid rmse-mean'])
bestscore = cv_results['valid rmse-mean'][-1]
print('best n_estimators:',best_estimator )
print('best cv score:', cv_results['valid rmse-mean'][-1])
end_time1 = time.time()
print("cose times:",end_time1-start_time1)

##---------- Step 2: max_dept and num_leaves ----------
start_time = time.time()
model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              num_leaves=50,
                              learning_rate=0.1,
                              n_estimators=best_estimator,
                              metric='rmse',
                              bagging_fraction=0.8,
                              feature_fraction=0.8,
                              subsample=None,
                              colsample_bytree=None)

params_test1={
'max_depth': range(2, 11, 3),
'num_leaves':range(10, 150, 30)
}

filtered_params = [
    param for param in ParameterGrid(params_test1) if param['num_leaves'] <= 2**param['max_depth']
]

param_grid = {
    'max_depth': sorted(set(p['max_depth'] for p in filtered_params)),  # 去重并排序
    'num_leaves': sorted(set(p['num_leaves'] for p in filtered_params))  # 去重并排序
}

gsearch1 = GridSearchCV(
    estimator=model_lgb,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

gsearch1.fit(x_train, y_train)
best_depth = gsearch1.best_params_["max_depth"]
best_leaves = gsearch1.best_params_["num_leaves"]
bestscore = gsearch1.best_score_
print("CV:", gsearch1.cv_results_)
print("best params:", gsearch1.best_params_)
print("best score:", gsearch1.best_score_)
end_time2 = time.time()
print("cose times:", end_time2 - start_time)

##--- Refinement Testing for max_depth and num_leaves ---
# Note: this can be added repeatedly
start_time = time.time()
model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              learning_rate=0.1,
                              n_estimators=best_estimator,
                              metric='rmse',
                              bagging_fraction=0.8,
                              feature_fraction=0.8)
params_test2 = {
    'max_depth': [best_depth - 1, best_depth, best_depth + 1],
    'num_leaves': [best_leaves - 1, best_leaves, best_leaves + 1],
}

gsearch2 = GridSearchCV(estimator=model_lgb,
                        param_grid=params_test2,
                        scoring='neg_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=-1)

gsearch2.fit(x_train, y_train)
best_depth = gsearch2.best_params_["max_depth"]
best_leaves = gsearch2.best_params_["num_leaves"]
print("CV:", gsearch2.cv_results_)
print("best params:", gsearch2.best_params_)
print("best score:", gsearch2.best_score_)
end_time3 = time.time()
print("cose times:", end_time3 - start_time)

# ##---------- setp 3: min_data_in_leaf and min_sum_hessian_in_leaf ----------
start_time = time.time()
params_test3={
    'min_child_samples': [5,10,15,20],
    'min_child_weight':[0.001,0.002,0.01,0.1]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              n_estimators=best_estimator,
                              num_leaves=best_leaves,
                              max_depth=best_depth,
                              metric='rmse',
                              bagging_fraction=0.8,
                              feature_fraction=0.8,
                              learning_rate=0.1,
                              subsample=None,
                              colsample_bytree=None)

gsearch3 = GridSearchCV(estimator=model_lgb,
                        param_grid=params_test3,
                        scoring='neg_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=-1)

gsearch3.fit(x_train, y_train)

best_child_samples = gsearch3.best_params_["min_child_samples"]
best_child_weight = gsearch3.best_params_["min_child_weight"]
bestscore = gsearch3.best_score_
print("CV:", gsearch3.cv_results_)
print("best params:", gsearch3.best_params_)
print("best score:", gsearch3.best_score_)
end_time4 = time.time()
print("cose times:", end_time4 - start_time)

best_child_samples = 5
best_child_weight = 0.001

##--- Refinement Testing for min_data_in_leaf and min_sum_hessian_in_leaf ---
# Note: this can be added repeatedly
start_time = time.time()
params_test3={
    'min_child_samples': [best_child_samples-2,best_child_samples-1,best_child_samples,best_child_samples+1,best_child_samples+2],
    'min_child_weight':[best_child_weight]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt',
                n_estimators = best_estimator,
                num_leaves = best_leaves,
                max_depth = best_depth,
                metric='rmse',
                learning_rate= 0.1,
                bagging_fraction = 0.8,
                feature_fraction = 0.8,)

gsearch3 = GridSearchCV(estimator=model_lgb,
             param_grid=params_test3,
             scoring='neg_mean_squared_error',
             cv=5,
             verbose=1,
             n_jobs=-1)

gsearch3.fit(x_train, y_train)

best_child_samples = gsearch3.best_params_["min_child_samples"]
print("CV:", gsearch3.cv_results_)
print("best params:",gsearch3.best_params_)
print("best score:", gsearch3.best_score_)
end_time5 = time.time()
print("cose times:",end_time5-start_time)

# ##---------- Step 4: feature_fraction and bagging_fraction ----------
start_time = time.time()
params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.8, 1.0]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              n_estimators=best_estimator,
                              num_leaves=best_leaves,
                              max_depth=best_depth,

                              min_child_samples=best_child_samples,
                              min_child_weight=best_child_weight,
                              metric='rmse',
                              learning_rate=0.1,
                              subsample=None,
                              colsample_bytree=None)

gsearch4 = GridSearchCV(estimator=model_lgb,
                        param_grid=params_test4,
                        scoring='neg_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=-1)

gsearch4.fit(x_train, y_train)

best_feature_fraction = gsearch4.best_params_["feature_fraction"]
best_bagging_fraction = gsearch4.best_params_["bagging_fraction"]
bestscore = gsearch4.best_score_
print("CV:", gsearch4.cv_results_)
print("best params:", gsearch4.best_params_)
print("best score:", gsearch4.best_score_)
end_time6 = time.time()
print("cose times:", end_time6 - start_time)

##--- Refinement Testing for feature_fraction ---
# Note: this can be added repeatedly
start_time = time.time()
params_test5 = {
    'feature_fraction': [best_feature_fraction - 0.1, best_feature_fraction - 0.05, best_feature_fraction - 0.02,
                         best_feature_fraction, best_feature_fraction + 0.02, best_feature_fraction + 0.05,
                         best_feature_fraction + 0.1]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              n_estimators=best_estimator,
                              num_leaves=best_leaves,
                              max_depth=best_depth,

                              min_child_samples=best_child_samples,
                              min_child_weight=best_child_weight,

                              bagging_fraction=best_bagging_fraction,
                              metric='rmse',
                              bagging_freq=5,
                              learning_rate=0.1,
                              )

gsearch5 = GridSearchCV(estimator=model_lgb,
                        param_grid=params_test5,
                        scoring='neg_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=4)

gsearch5.fit(x_train, y_train)

best_feature_fraction = gsearch5.best_params_["feature_fraction"]
print("CV:", gsearch5.cv_results_)
print("best params:", gsearch5.best_params_)
print("best score:", gsearch5.best_score_)
end_time7 = time.time()
print("cose times:", end_time7 - start_time)

# ##---------- step 5: learning_rate ----------
start_time = time.time()
params_test5={
    'learning_rate': [0.1,0.001,0.002,0.005,0.008]
}
model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              n_estimators=best_estimator,
                              num_leaves=best_leaves,
                              max_depth=best_depth,
                              min_child_samples=best_child_samples,
                              min_child_weight=best_child_weight,
                              bagging_fraction=best_bagging_fraction,
                              feature_fraction=best_feature_fraction,
                              metric='rmse',
                              subsample=None,
                              colsample_bytree=None
                              )

gsearch5 = GridSearchCV(estimator=model_lgb,
                        param_grid=params_test5,
                        scoring='neg_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=4)

gsearch5.fit(x_train, y_train)

best_learning_rate = gsearch5.best_params_["learning_rate"]
bestscore = gsearch5.best_score_
print("CV:", gsearch5.cv_results_)
print("best params:", gsearch5.best_params_)
print("best score:", gsearch5.best_score_)
end_time8 = time.time()
print("cose times:", end_time8 - start_time)

#---------- setp 6:  n_estimators ----------
start_time = time.time()
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate':best_learning_rate,
    'num_leaves':best_leaves,
    'max_depth':best_depth,
    'min_child_samples':best_child_samples,
    'min_child_weight':best_child_weight,
    'bagging_fraction':best_bagging_fraction,
    'feature_fraction':best_feature_fraction,
    }


data_train = lgb.Dataset(x_train, y_train, silent=True)

cv_results = lgb.cv(
        params,
        data_train,
        num_boost_round=50000,
        nfold=5,
        stratified=False,
        shuffle=True,
        metrics='rmse',
        callbacks=[
            lgb.early_stopping(500),
            lgb.log_evaluation(period=100)
        ]
        seed=0)


best_estimator = len(cv_results['rmse-mean'])
print('best n_estimators:',best_estimator )
print('best cv score:', cv_results['rmse-mean'][-1])
bestscore = cv_results['rmse-mean'][-1]
end_time1 = time.time()
print("cose times:",end_time1-start_time)

print("best_estimator:", best_estimator)
print("best_leaves:", best_leaves)
print("best_depth:", best_depth)
print("best_child_samples:", best_child_samples)
print("best_child_weight:", best_child_weight)
print("best_bagging_fraction:", best_bagging_fraction)
print("best_feature_fraction:", best_feature_fraction)
print("best_learning_rate:", best_learning_rate)
end_time9 = time.time()
print("cose times:", end_time9 - start_time)

#%% [Step 3-4: Train LGBM and Accuracy evaluation]
# In[By lightgbm (lgb)]
import lightgbm as lgb
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

year = 2003
Y_col = "AVP" #['AVP', 'DPT', 'MR', 'RH', 'SH', 'VPD']

###------ Load data ------
Train_path = r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Trainsample.csv"
Train = pd.read_csv(Train_path)
Train_cleaned = Train.drop(columns=Train.filter(regex="Unname").columns)
Train = Train_cleaned[~Train_cleaned.isin([-9999]).any(axis=1)]
Train['year'] = pd.to_datetime(Train['time'], errors='coerce').dt.year

Test_path = r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Testsample.csv"
Test = pd.read_csv(Test_path)
Test_cleaned = Test.drop(columns=Test.filter(regex="Unname").columns)
Test = Test_cleaned[~Test_cleaned.isin([-9999]).any(axis=1)]
Test['year'] = pd.to_datetime(Test['time'], errors='coerce').dt.year
Test['month'] = pd.to_datetime(Test['time'], errors='coerce').dt.month
Test['day'] = pd.to_datetime(Test['time'], errors='coerce').dt.day


data = Train_Ayear.iloc[:, 1:8]
if Y_col = "AVP":
    data['avp'] = Train_Ayear.iloc[:, 8]
if Y_col = "DPT":
    data['dpt'] = Train_Ayear.iloc[:, 9]
if Y_col = "MR":
    data['mr'] = Train_Ayear.iloc[:, 10]
if Y_col = "RH":
    data['rh'] = Train_Ayear.iloc[:, 11]
if Y_col = "SH":
    data['sh'] = Train_Ayear.iloc[:, 12]
if Y_col = "VPD":
    data['vpd'] = Train_Ayear.iloc[:, 13]
data[Y_col] = Train_Ayear[Y_col]
x_train = data.iloc[:, 0:8]
y_train = data[Y_col]

data1 = Test_Ayear.iloc[:, 1:8]
if Y_col = "AVP":
    data1['avp'] = Test_Ayear.iloc[:, 8]
if Y_col = "DPT":
    data1['dpt'] = Test_Ayear.iloc[:, 9]
if Y_col = "MR":
    data1['mr'] = Test_Ayear.iloc[:, 10]
if Y_col = "RH":
    data1['rh'] = Test_Ayear.iloc[:, 11]
if Y_col = "SH":
    data1['sh'] = Test_Ayear.iloc[:, 12]
if Y_col = "VPD":
    data1['vpd'] = Test_Ayear.iloc[:, 13]
data1[Y_col] = Test_Ayear[Y_col]
x_test = data1.iloc[:, 0:8]
y_tes = data1[Y_col]

###------ Train ------
best_estimator = 100000
best_learning_rate = 0.05
best_depth = 9
best_leaves = 20
best_child_samples = 90
best_child_weight= 0.001
best_bagging_fraction = 0.8
best_feature_fraction = 0.6

LightGBM_model = lgb.LGBMRegressor(
    n_estimators=best_estimator,
    num_leaves=best_leaves,
    max_depth=best_depth,
    min_child_samples=best_child_samples,
    min_child_weight=best_child_weight,
    bagging_fraction=best_bagging_fraction,
    feature_fraction=best_feature_fraction,
    learning_rate=best_learning_rate,

    verbosity=100,
    boosting_type='gbdt',
    objective='regression',
    importance_type="gain",
    n_jobs=-1,
    random_state=42
)

LightGBM_model.fit(x_train.values,y_train.values)

###------ Save Model ------
booster = LightGBM_model.booster_
booster.save_model(r'E:\high-resolution atmospheric moisture\Train\model\trained_LGBM_model_{}_{}.txt'.format(Y_col, year))

###------ Predict ------
y_pred = LightGBM_model.predict(x_test.values)

###------ Accuracy Assessment ------
if 'y_test' in locals():
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
Test_Ayear['id'] = Test_Ayear['station']
Result_data = pd.DataFrame()
Result_data["id"] = Test_Ayear["id"]
Result_data['lon'] = np.nan
Result_data['lat'] = np.nan
Result_data["year"] = Test_Ayear["year"]
Result_data["month"] = Test_Ayear["month"]
Result_data["day"] = Test_Ayear["day"]
Result_data["y_test"] = y_test
###------ load point data including lon/lat ------
Point_Lonlat_path = r"E:\high-resolution atmospheric moisture\Data\station_info.csv"  #column: id, lon, lat
Point_Lonlat = pd.read_csv(Point_Lonlat_path)
ss = Point_Lonlat.filter(regex="Unname")
Point_Lonlat = Point_Lonlat.drop(ss, axis=1)
Point_Lonlat = Point_Lonlat.sort_values(by=['id'], ascending=[True])
for num in range(Result_data.shape[0]):
    Aid = Result_data[num:num + 1]["id"].values[0]
    Arowdata = Point_Lonlat.loc[Point_Lonlat['id'].isin([Aid])]
    Result_data.iloc[num, 1] = Arowdata["lon"].values[0]
    Result_data.iloc[num, 2] = Arowdata["lat"].values[0]
Result_data["LGBM_y_pred"] = y_pred
Result_data["difference"] = Result_data["LGBM_y_pred"]-Result_data["y_test"]

file_path = f"E:\\high-resolution atmospheric moisture\\Train\\{year}\\{Y_col}_{year}y_test_data.csv"
Result_data.to_csv(file_path, index=False)

#%% [Step 5: Prodicte HPT - LGBM]
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
import time
import lightgbm as lgb
from sklearn import metrics
from osgeo import gdal
import xgboost as xgb

# In[Functions]
# seek filename in specific folder
def bseek(bootdir, Ftype, Alist):
    import os
    sfds = os.listdir(bootdir)  # search under the specific folder
    sfds.sort()
    for sfd in sfds:
        s = "/".join((bootdir, sfd))
        if os.path.isdir(s):
            bseek(s, Ftype, Alist)
        elif os.path.isfile(s):
            Alist.append(s)


def check_leap(input_year):
    if ((input_year % 400 == 0) | (input_year % 4 == 0)):
        return 1
    else:
        return 0


# %% [Step 1: Input data]
# In[Load data]

###------ Load Training and Testing Data ------
Train_path = r"/public/xml/High/Split Data/2003/Trainsample.csv"
Train = pd.read_csv(Train_path)
Train_cleaned = Train.drop(columns=Train.filter(regex="Unname").columns)
Train_cleaned = Train_cleaned[~Train_cleaned.isin([-9999]).any(axis=1)]
Train = Train_cleaned

Test_path = r"/public/xml/High/Split Data/2003/humidity/Testsample.csv"
Test = pd.read_csv(Test_path)
Test_cleaned = Test.drop(columns=Test.filter(regex="Unname").columns)
Test_cleaned = Test_cleaned[~Test_cleaned.isin([-9999]).any(axis=1)]
Test = Test_cleaned.dropna()

Train_Ayear = Train.loc[Train['year'].isin([year])]
Test_Ayear = Test.loc[Test['year'].isin([year])]

###------ load point data including lon/lat ------
Point_Lonlat_path = r"/public/xml/High/Split Data/point.csv"
Point_Lonlat = pd.read_csv(Point_Lonlat_path)
ss = Point_Lonlat.filter(regex="Unname")
Point_Lonlat = Point_Lonlat.drop(ss, axis=1)
Point_Lonlat = Point_Lonlat.sort_values(by=['id'], ascending=[True])
del ss, Point_Lonlat_path

# In[Chose data ]
# Note:Modeling each index year by year
###------ Choose the Index waiting for training ------
year = 2003  # from 2003 to 2020
Y_col = "RH" # indices including ['AVP', 'DPT', 'MR', 'RH', 'SH', 'VPD']
data = Train_Ayear.iloc[:, 1:8]
if Y_col = "AVP":
    data['avp'] = Train_Ayear.iloc[:, 8]
if Y_col = "DPT":
    data['dpt'] = Train_Ayear.iloc[:, 9]
if Y_col = "MR":
    data['mr'] = Train_Ayear.iloc[:, 10]
if Y_col = "RH":
    data['rh'] = Train_Ayear.iloc[:, 11]
if Y_col = "SH":
    data['sh'] = Train_Ayear.iloc[:, 12]
if Y_col = "VPD":
    data['vpd'] = Train_Ayear.iloc[:, 13]
data[Y_col] = Train_Ayear[Y_col]
x_train = data.iloc[:, 0:8]
y_train = data[Y_col]


Test_Ayear = Test.loc[Test['year'].isin([year])]
data1 = Test_Ayear.iloc[:, 1:8]
if Y_col = "AVP":
    data1['avp'] = Test_Ayear.iloc[:, 8]
if Y_col = "DPT":
    data1['dpt'] = Test_Ayear.iloc[:, 9]
if Y_col = "MR":
    data1['mr'] = Test_Ayear.iloc[:, 10]
if Y_col = "RH":
    data1['rh'] = Test_Ayear.iloc[:, 11]
if Y_col = "SH":
    data1['sh'] = Test_Ayear.iloc[:, 12]
if Y_col = "VPD":
    data1['vpd'] = Test_Ayear.iloc[:, 13]
data1[Y_col] = Test_Ayear[Y_col]
data1 = data1.dropna()
x_test = data1.iloc[:, 0:8]
y_test = data1[Y_col]


###------ Dataframe for saving results ------
Result_data = pd.DataFrame()
Result_data["id"] = Test_Ayear["id"]
Result_data['lon'] = np.nan
Result_data['lat'] = np.nan
Result_data["year"] = Test_Ayear["year"]
Result_data["month"] = Test_Ayear["month"]
Result_data["day"] = Test_Ayear["day"]
Result_data["y_test"] = y_test

# 根据 id 从 Point_Lonlat 中获取经纬度数据
for num in range(Result_data.shape[0]):
    # num = 0
    Aid = Result_data[num:num + 1]["id"].values[0]
    Arowdata = Point_Lonlat.loc[Point_Lonlat['id'].isin([Aid])]
    Result_data.iloc[num, 1] = Arowdata["lon"].values[0]
    Result_data.iloc[num, 2] = Arowdata["lat"].values[0]

# 直接加载模型
LightGBM_model = lgb.Booster(model_file='/public/xml/High/model/2003/trained_LGBM_model_RH.txt')
print(type(LightGBM_model))


###------ seek original image and check saving path ------
#--- Retrieve the whole year original image ---
Alist = [] #saving path
bootpath = r"/public/xml/High/3 Predict Data/{}/".format(year)
bseek(bootpath,"tif",Alist)

#preparation
OutputPath_root = r"/public/xml/High/Predict/{}/".format(year)
if not os.path.exists(OutputPath_root):
        os.mkdir(OutputPath_root)
OutputPath_root = OutputPath_root + "{}/".format(Y_col)
if not os.path.exists(OutputPath_root):
        os.mkdir(OutputPath_root)

#Run
regsor = "_LGBM"  #name suffix 命名输出文件时的后缀
print('{} start:'.format(regsor))
startTime11 = time.time()
for num in range(len(Alist)):
    img_meta = gdal.Open(Alist[num])
    if Y_col = "AVP":
        selected_indices = list(range(1, 8)) + [9]
    if Y_col = "DPT":
        selected_indices = list(range(1, 8)) + [11]
    if Y_col = "MR":
        selected_indices = list(range(1, 8)) + [12]
    if Y_col = "RH":
        selected_indices = list(range(1, 8)) + [8]
    if Y_col = "SH":
        selected_indices = list(range(1, 8)) + [13]
    if Y_col = "VPD":
        selected_indices = list(range(1, 8)) + [10]

    selected_bands = []

    for idx in selected_indices:
        band = img_meta.GetRasterBand(idx)
        band_array = band.ReadAsArray()
        selected_bands.append(band_array)

    img = np.stack(selected_bands)
    new_shape = (img.shape[0],-1)
    img_as_array = img[:,:,:].reshape(new_shape).T

    #Mask
    mask = pd.DataFrame(img_as_array)!= -9999.0
    final_mask = mask[0]
    for col in mask.columns:
        final_mask = np.bitwise_and(final_mask,mask[col])

    #Fill the null with -9999
    from sklearn.impute import SimpleImputer
    img_as_array[np.isnan(img_as_array)] = -9999.0
    imputer = SimpleImputer(missing_values = -9999.0, strategy = 'constant', fill_value = 0)
    mask_all = imputer.fit_transform(img_as_array)
    mask_all = pd.DataFrame(mask_all)
    mask_all.columns = x_test.columns

    #predict
    img_pred = LightGBM_model.predict(mask_all)

    img_pred = img_pred.reshape(img[1,:,:].shape)
    final_mask = final_mask.values.reshape(img[1,:,:].shape)
    if Y_col != "DPT":
        img_pred = np.clip(img_pred, 0, 100)

    #adjust mask
    import numpy.ma as ma
    img_pred = img_pred * final_mask
    masked_img_pred = ma.masked_array(data=img_pred, mask=~final_mask)
    filled_masked_img_pred = masked_img_pred.filled(-9999.0)

    #output image
    #--- set name rule ---
    print(Alist[num])
    OutputPath_1 = OutputPath_root + Alist[num].split("/")[5]
    if not os.path.exists(OutputPath_1):
        os.mkdir(OutputPath_1)

    output_filename = OutputPath_1 + "/" + Alist[num].split("/")[-1].split(".")[0] + regsor + ".tif"

    x_pixels = img_pred.shape[1]
    y_pixels = img_pred.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_filename, x_pixels, y_pixels, 1, gdal.GDT_Float32)

    #Add geographic coordinates and projections
    geotrans = img_meta.GetGeoTransform()
    proj = img_meta.GetProjection()
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    dataset.GetRasterBand(1).WriteArray(filled_masked_img_pred)
    dataset.GetRasterBand(1).SetNoDataValue(-9999.0)
    dataset.FlushCache()
    dataset = None
    print('--- {} has Done! ---'.format(Alist[num].split("/")[-1].split(".")[0]))

startTime12 = time.time()
print("%f"%(startTime12-startTime11))
