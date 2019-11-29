import numpy as np
import pandas as pd
from joblib import load
import os
import json
from flask import Flask
from flask import request
import collections
from datetime import datetime
import tensorflow as tf

# from google.appengine.api import app_identity
app = Flask(__name__)

@app.route("/")
def hello():
    print('blaaaaaaaaaah')
    startTime = datetime.now()

    resultmap = {}
    resultmap['prediction'] = 'Leroy'
    endtime = datetime.now() - startTime
    endtime = str(endtime.microseconds)
    resultmap['exec_time'] = endtime;

    # print(type(endtime))

    # print(resultmap)

    return json.dumps(resultmap)
    # return "Hello World!"

@app.route('/test', methods=['POST'])
def test():
    startTime = datetime.now()
    print('yeahhhhh')
    # print(type(request.json))

    df = json.loads(request.data)

    df = pd.DataFrame(df)

    # print(df)
    # print(df.shape)

    
    col_drops = [ x for x in df.index if int(x)%2 ==1 ]
    print(len(col_drops), ' ass')
    df.drop(col_drops, axis=0, inplace=True)

    df = df.values.T
    df = np.expand_dims(df, axis=2)
    print('ss', df.shape)


    # print('df shape::',df.shape)

    # df = centralize(df)
    # aggdf = data_aggregate(df)
    # # print('aggdf shape::',aggdf.shape)

    # models = load_models()

    # # print(aggdf.head())

    # aggdf_fs = models['Feture Selection'].transform(aggdf)
    # print('aggdf_fs shape ', aggdf_fs.shape)

    # labelMapping = {}
    # labelMapping[0] = 'buy'
    # labelMapping[1] = 'communicate'
    # labelMapping[2] = 'fun'
    # labelMapping[3] = 'hope'
    # labelMapping[4] = 'mother'
    # labelMapping[5] = 'really'

    new_model = tf.keras.models.load_model('lstm_eeg.h5')

    # Check its architecture
    new_model.summary()

    resultmap = {}

    # i=1
    # for modelname, model in models.items():
    #     res = model.predict(aggdf_fs)
    #     print('result of ',modelname , '::' ,res)
    #     resultmap['prediction'] = labelMapping[res[0]]
    #     i=i+1


    print(new_model.predict(df))

    resultmap['prediction'] = 'Leroy'
    endtime = datetime.now() - startTime
    endtime = str(endtime.microseconds)
    resultmap['exec_time'] = endtime;

    print(resultmap)

    return json.dumps(resultmap)


MODEL_FILE_PATH = './'
MODEL_FILENAMES = {
    # 'Gaussian Naive Bayes' : 'gnb_clf.joblib',
    'xgb' : 'xgb_clf.joblib',
    'Logistic Regression' : 'lr_clf.joblib',
    'Random Forest' : 'rf_clf.joblib',
    'Voting Classifier' : 'vot_clf.joblib',
    'Feture Selection' : 'feat_selection.joblib'
}

def load_models():
    models = {}
    for m in MODEL_FILENAMES:
        mod = load(MODEL_FILE_PATH+MODEL_FILENAMES[m])
        models[m] = mod
    return models

def centralize(df):
    x_columns = [x for x in df.columns if '_x' in x]
    x_shift = df[["rightShoulder_x", "leftShoulder_x"]].sum(axis=1)/2
    for col in x_columns:
        df[col] = df[col] - x_shift
            
    y_columns = [y for y in df.columns if '_y' in y]
    y_shift = df[["rightShoulder_y", "leftShoulder_y", "leftHip_y", "rightHip_y"]].sum(axis=1)/4
    for col in x_columns:
        df[col] = df[col] - y_shift

    return df


def data_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # skip_cols = ['Frames#']
    df['vid_id'] = 1

    # agg_data = df.groupby('vid_id').agg([np.mean, np.std])

    agg_data = df.groupby('vid_id').agg([np.mean, np.std, np.min, np.max, pd.DataFrame.kurt, pd.DataFrame.skew])
    agg_data.columns = ["_".join(x) for x in agg_data.columns.ravel()]
    agg_data = agg_data.reset_index()
    agg_data.drop(['vid_id'], axis = 1, inplace=True)

    return agg_data


if __name__ == '__main__':

    app.run(host='127.0.0.1', port=80, debug=True)

# if __name__ == '__main__':
#     app.run(host='192.168.0.7', port=80, debug=True)

# 127.0.0.1
    # 192.168.0.7

# app.run(port=80)