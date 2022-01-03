from flask import Flask
from flask import request
import pandas as pd
import pickle
import numpy as np

churn_model = pickle.load(open('churn_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/predict_churn')
def predict_churn():
    features_dic = {'is_male': request.args.get('is_male'),
                    'num_inters': request.args.get('num_inters'),
                    'late_on_payment': request.args.get('late_on_payment'),
                    'age': request.args.get('age'),
                    'years_in_contract': request.args.get('years_in_contract')
                    }

    X_new = np.fromiter(features_dic.values(), dtype=float)
    predict_me = X_new.reshape(1, -1)
    y_p = churn_model.predict(predict_me)[0]

    return f'{y_p}'


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=8080)
