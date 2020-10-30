import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor

def rmse(true,pred):
    res =  mean_squared_error(np.log(true), np.log(pred), squared=False)
    return res

def train(model, X, y, grid, metric, metric_module, greater):
    
    # Preprocess

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
    grid_search = GridSearchCV(model, 
                            param_grid=grid,
                            scoring=make_scorer(metric_module, greater_is_better=greater),
                            cv=5)
    grid_search.fit(X_train,y_train)
    
    valid_score = grid_search.best_score_
    test_score = metric_module(y_test, grid_search.best_estimator_.predict(X_test))
    print(f'validation_{metric}:{valid_score }' )
    print(f'test_{metric}:{test_score}' )
    
    return grid_search, valid_score, test_score

if __name__ == '__main__':

    # Data Loading

    # debug:: check the file inside this folder if the input is working
    dirs = os.listdir("/opt/ml/input/data/training/")
    for d in dirs:
        print("data file input: ")
        print(d)



    df = pd.read_csv("/opt/ml/input/data/training/train.csv")
    print("Data shape : {}".format(df.shape))

    features = [x for x in df.columns if x not in ['SalePrice']]
    X = df[features]
    y = df['SalePrice']

    numerical_cols = [cname for cname in X.columns if 
                X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in X.columns if
                        X[cname].nunique() < 13 and 
                        X[cname].dtype == "object"]


    numerical_transformer = SimpleImputer(strategy='constant')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
    ])

    X = preprocessor.fit_transform(X)

    xgb_model = XGBRegressor()
    params = {
            'n_estimators': [500, 1000],
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
            }
    xgb_model_clf, valid_score, test_score= train(xgb_model, X, y, params, "rmse", rmse, greater=False) 
    
    # Creating Report for Output
    
    metric  = {
    'valid_score': valid_score,
    'test_score':test_score,
    'best_params':xgb_model_clf.best_params_
    }
    

    model_output_path = "/opt/ml/model/model.pkl"
    metric_path = "/opt/ml/model/metric.json"

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        joblib.dump(xgb_model_clf,f)
    with open(metric_path, 'w') as f:
        metric_json = json.dumps(metric, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        f.write(metric_json)


