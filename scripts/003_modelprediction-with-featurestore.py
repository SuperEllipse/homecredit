import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import ELU, PReLU, LeakyReLU
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import joblib
import os
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import keras as keras
from feast import FeatureStore

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# df = pd.read_csv('/home/cdsw/04_RAW_TRAINING_DATA/processed_data.csv')
# print("Raw shape: ", df.shape)



store = FeatureStore(repo_path='/home/cdsw/homecredit_feature_repo')

# We have saved the test data with a timestamp of 2024
entity_info = f"""
            SELECT sk_id_curr, current_timestamp as event_timestamp FROM  homecredit_processed_data 
            WHERE event_timestamp > '2024-01-01'    
            """
#training_df = store.get_historical_features(
#    entity_df = entity_info,
#    features=[
#         "homecredit_appl_details_view:target",
#         "homecredit_appl_details_view:flag_own_car",
#         "homecredit_appl_details_view:flag_own_realty",
#    ]
#).to_df()

feature_service = store.get_feature_service("homecredit_appl_details_Fservice")
df=store.get_historical_features(features=feature_service, entity_df = entity_info).to_df()
print("Raw shape: ", df.shape)

print("----- Example features -----\n")
print(df.head(3))



nn = keras.models.load_model('/home/cdsw/06_ML_MODEL/nn_model')
#
#df = pd.read_csv('/home/cdsw/02_PREDICTION_DATA/processed_data.csv')
#print("Raw shape: ", df.shape)

y = df['target']
feats = [f for f in df.columns if f not in ['target','sk_id_curr','sk_id_bureau','sk_id_prev','index', 'event_timestamp', 'created']]
X = df[feats]
print("X shape: ", X.shape)

print("\nPreparing data...")

X = X.fillna(X.mean()).clip(-1e11,1e11)
scaler = MinMaxScaler()
scaler.fit(X)

nn_predict=nn.predict(X).flatten().clip(0,1)

print( 'Saving results...' )
nn_predict_result = pd.DataFrame()
nn_predict_result['sk_id_curr'] = df['sk_id_curr']
nn_predict_result['target'] = nn_predict


nn_predict_result.to_csv('/home/cdsw/03_PREDICTION_RESULT/nn_predict_result.csv')