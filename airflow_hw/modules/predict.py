import glob
import json
import os

import dill
import pandas
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    with open(f'{path}/data/models/cars_pipe_202403100032.pkl', 'rb') as file:
        model = dill.load(file)

    test_files = glob.glob(f'{path}/data/test/*.json')
    test_list = []

    for test_file in test_files:
        with open(test_file) as fin:
            test_list.append(json.load(fin))

    df_pred = pandas.DataFrame.from_dict(test_list)
    df_pred['preds'] = model.predict(df_pred)
    df_pred.to_csv(f'{path}/data/predictions/'
                   f'preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
