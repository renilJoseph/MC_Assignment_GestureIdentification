import numpy as np
import pandas as pd
from joblib import load
import os
import json
from flask import Flask
from flask import request
import collections

# from google.appengine.api import app_identity
app = Flask(__name__)

@app.route("/")
def hello():
    print(request.args)
    return "Hello World!"

@app.route('/test', methods=['POST'])
def test():
    print('yeahhhhh')
    # print(type(request.json))

    df = convert_to_csv(json.loads(request.data))

    # print(df.head())
    # print('df shape::',df.shape)

    df = centralize(df)
    aggdf = data_aggregate(df)
    # print('aggdf shape::',aggdf.shape)

    models = load_models()

    # print(aggdf.head())

    aggdf_fs = models['Feture Selection'].transform(aggdf)
    # print('aggdf_fs shape ', aggdf_fs.shape)

    labelMapping = {}
    labelMapping[0] = 'buy'
    labelMapping[1] = 'communicate'
    labelMapping[2] = 'fun'
    labelMapping[3] = 'hope'
    labelMapping[4] = 'mother'
    labelMapping[5] = 'really'

    resultmap = {}
    i=1
    for modelname, model in models.items():
        if i == 5:
            break
        res = model.predict(aggdf_fs)
        print('result of ',modelname , '::' ,res)
        resultmap[i] = labelMapping[res[0]]
        i=i+1

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

def convert_to_csv(data):
    ncol = set()
    tmpmap = collections.OrderedDict()
    tmplist = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 
    'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 
    'leftAnkle', 'rightAnkle']

    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        # print(data[i])
        one.append(data[i]['score'])
        # print('lengt :: ',len(data[i]['keypoints']))
        for x in tmplist:
            one = [0, 0, 0]
            tmpmap[x] = one
        for obj in data[i]['keypoints']:
            one = []
            if 'score' in obj:
                one.append(obj['score'])
            else:
                one.append(0.0)
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
            if 'part' in obj:
                tmpmap[obj['part']] = one
            else:
                print('not founf ',one)
        one = []
        one.append(data[i]['score'])
        for key, val in tmpmap.items():
            one.extend(val)
        csv_data[i] = np.array(one)
    df = pd.DataFrame(csv_data, columns=columns)
    return df


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80, debug=True)

# app.run(port=80)