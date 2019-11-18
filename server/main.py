import numpy as np
import pandas as pd
from joblib import load
import os
from flask import Flask
from flask import request

# from google.appengine.api import app_identity
app = Flask(__name__)

@app.route("/")
def hello():
    print(request.args)
    return "Hello World!"

@app.route('/test', methods=['POST'])
def test():
    print('yeahhhhh')
    print(request.args)
    # imagefile = request.files['media']
    # bytes = imagefile.read()
    return request.args

if __name__ == '__main__':
#     # This is used when running locally only. When deploying to Google App
#     # Engine, a webserver process such as Gunicorn will serve the app. This
#     # can be configured by adding an `entrypoint` to app.yaml.
#     # Flask's development server will automatically serve static files in
#     # the "static" directory. See:
#     # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
#     # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=80, debug=True)


MODEL_FILE_PATH = '/Users/renil.joseph/Documents/github/class/mcCode/assign2/models/'
MODEL_FILENAMES = {
    'Gaussian Naive Bayes' : 'gnb_clf.joblib',
    'Logistic Regression' : 'lr_clf.joblib',
    'Random Forest' : 'rf_clf.joblib',
    'Voting Classifier' : 'vot_clf.joblib'
}

def load_models():
    models = {}
    for m in MODEL_FILENAMES:
        mod = load(MODEL_FILE_PATH+MODEL_FILENAMES[m])
        models.update()
    return None


def data_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    skip_cols = ['Frames#']

    agg_data = df.drop(skip_cols, axis=1).groupby('vid_id').agg([np.mean, np.std])
    agg_data.columns = ["_".join(x) for x in agg_data.columns.ravel()]
    agg_data = agg_data.reset_index()

    return agg_data

# app.run()