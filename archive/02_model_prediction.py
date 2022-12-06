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
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras as keras

nn = keras.models.load_model('/home/cdsw/06_ML_MODEL/nn_model')

df = pd.read_csv('/home/cdsw/02_PREDICTION_DATA/processed_data.csv')
print("Raw shape: ", df.shape)

y = df['TARGET']
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
X = df[feats]
print("X shape: ", X.shape)

print("\nPreparing data...")

X = X.fillna(X.mean()).clip(-1e11,1e11)
scaler = MinMaxScaler()
scaler.fit(X)

nn_predict=nn.predict(X).flatten().clip(0,1)

print( 'Saving results...' )
nn_predict_result = pd.DataFrame()
nn_predict_result['SK_ID_CURR'] = df['SK_ID_CURR']
nn_predict_result['TARGET'] = nn_predict


nn_predict_result.to_csv('/home/cdsw/03_PREDICTION_RESULT/nn_predict_result.csv')