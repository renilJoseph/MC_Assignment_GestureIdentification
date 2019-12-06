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
    eeg_model = tf.keras.models.load_model('lstm_eeg.h5')
    startTime = datetime.now()
    print('yeahhhhh')
    # print(type(request.json))
    #print(request.data)

    df = json.loads(request.data)

    df = pd.DataFrame(np.fromstring(df['data'][1:-1], dtype=float, sep=','))

    # print(df)
    # print(df.shape)

    
    col_drops = [ x for x in df.index if int(x)%2 ==1 ]
#    print(len(col_drops), ' ass')
    df.drop(col_drops, axis=0, inplace=True)

    df = df.values.T
    df = np.expand_dims(df, axis=2)
    print('ss', df.shape)
    

    # Check its architecture
    #new_model.summary()

    resultmap = {}



    prediction = np.argmax(eeg_model.predict(df)[0])


    resultmap['prediction'] = str(prediction)
    endtime = datetime.now() - startTime
    endtime = str(endtime.microseconds/1000)
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


# //for cloud
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)

# //for fog
# if __name__ == '__main__':
#     app.run(host='192.168.0.7', port=80, debug=True)

# 127.0.0.1
    # 192.168.0.7

# app.run()