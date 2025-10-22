from sklearn.model_selection import train_test_split , RandomizedSearchCV as rscv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import lightgbm as lgbm
import pandas as pd 
from imblearn.over_sampling import SMOTE
import numpy as np 
import joblib as jb 


df = pd.read_csv('fraud_dec_2.csv', nrows = 1000000)
df_copy = df.copy() 

#encoding 
cat_cols = ['type', 'nameOrig', 'nameDest']
for i in cat_cols:
    freq = df_copy[i].value_counts() / len(df_copy)
    df_copy[i] = df_copy[i].map(freq)
print(df_copy[cat_cols].value_counts())    
'''
#setting up x and y 
x = df_copy.drop(columns  = ['isFraud', 'isFlaggedFraud'],axis = 1)
y = df_copy['isFraud']

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 42)


a = SMOTE(random_state = 42)
x_train_res, y_train_res = a.fit_resample(x_train, y_train)


#def 
def long_data(x_train_res, y_train_res, x_test, y_test):
    model = lgbm.LGBMClassifier(   objective = 'binary',
    num_leaves = 67,
    n_estimators = 1000,
    max_depth = 5, 
    learning_rate =0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_samples = 40,
    scale_pos_weight = 3, 
    random_state = 42, class_weight = 'balanced', n_jobs = -1)
    model.fit(x_train_res, y_train_res, eval_set = [(x_test, y_test)], eval_metric = 'f1')
    pred = model.predict_proba(x_test)[:,1]
    precisions, _, thresholds = precision_recall_curve(y_test, pred)
    desired_precision = 0.7
    best_idx = np.argmax(precisions >= desired_precision)
    best_thresholds = thresholds[best_idx]
    y_pred = (pred > best_thresholds).astype(int)
    print('acc', accuracy_score(y_test, y_pred))
    print('prec', precision_score(y_test, y_pred))
    print('recall', recall_score(y_test, y_pred))
    print('f1', f1_score(y_test, y_pred))
    print('conf' , confusion_matrix(y_test, y_pred))
    return x_train_res, y_train_res, model, y_pred

halle = long_data(x_train_res, y_train_res, x_test, y_test)

jb.dump(halle, 'model.pkl')'''