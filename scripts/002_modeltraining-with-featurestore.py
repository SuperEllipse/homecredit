# This is just a skeleton.
# The NN is copied from an unrelated kernel with no attempt to adapt it to the use case,
#   except where absolutely necessary.
# The data preparation is the minimum necessary to get it to run OK.
# The input features are pre-engineered,
#   so this doesn't embody the idea that "neural networks create their own features."
# There is no real validation, except what Keras does.
# But it's a starting point.

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
import warnings
from datetime import datetime
import pandas as pd
from feast import FeatureStore

warnings.simplefilter(action='ignore', category=FutureWarning)

# df = pd.read_csv('/home/cdsw/04_RAW_TRAINING_DATA/processed_data.csv')
# print("Raw shape: ", df.shape)



store = FeatureStore(repo_path='/home/cdsw/homecredit_feature_repo')

entity_info = f"""
            SELECT sk_id_curr, current_timestamp as event_timestamp FROM  homecredit_processed_data 
            WHERE event_timestamp < '2024-01-01'    
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
print("----- Example features -----\n")
print(df.head(3))



y = df['target']
feats = [f for f in df.columns if f not in ['target','sk_id_curr','sk_id_bureau','sk_id_prev','index', 'event_timestamp', 'created']]
X = df[feats]
print("X shape: ", X.shape, "    y shape:", y.shape)

print("\nPreparing data...")
X = X.fillna(X.mean()).clip(-1e11,1e11)
scaler = MinMaxScaler()
scaler.fit(X)
training = y.notnull()
testing = y.isnull()
X_train = scaler.transform(X[training])
X_test = scaler.transform(X[testing])
y_train = np.array(y[training])
y_test = np.array(y[testing])
print( X_train.shape, X_test.shape, y_train.shape )

print( 'Setting up neural network...' )
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = 718))
nn.add(PReLU())
nn.add(Dropout(.3))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units = 12, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
nn.compile(loss='binary_crossentropy', optimizer='adam')

print( 'Fitting neural network...' )
nn.fit(X_train, y_train, validation_split=0.1, epochs=2, verbose=2)

print( 'Predicting...' )
y_pred = nn.predict(X_test).flatten().clip(0,1)

print( 'Saving results...' )
sub = pd.DataFrame()
sub['sk_id_curr'] = df[testing]['sk_id_curr']
sub['target'] = y_pred

print(sub.head() )
nn.save('/home/cdsw/06_ML_MODEL/nn_model')
