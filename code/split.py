import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def read_data():
    data = pd.read_csv('../output_files/data.csv')

    # categorical columns
    obj_colnames = ['homeown', 'income', 'race', 'lif_cyc', 'urbrur', 'htppopdn']
    for col in obj_colnames:
        data[col] = data[col].astype('object')
    return data


def const_x_y(data):
    # construct X and y
    y = data.iloc[:, data.columns == 'fuel']
    X = data.loc[:, ~data.columns.isin(['houseid', 'fuel'])]

    # dummy variables for categorical columns
    X_cat = X.loc[:, ~X.columns.isin(['vehmiles', 'hhvehcnt', 'hhsize', 'numadlt', 'drvrcnt', 'wrkcount'])]
    X_cat = pd.get_dummies(X_cat, dummy_na=True)

    # impute with mean numerical columns
    X_num = X.loc[:, X.columns.isin(['vehmiles', 'hhvehcnt', 'hhsize', 'numadlt', 'drvrcnt', 'wrkcount'])]
    num_col = X_num.columns
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_num = imp.fit_transform(X_num)
    X_num = pd.DataFrame(X_num)
    X_num.columns = num_col

    # construct X again
    X_final = pd.concat([X_cat, X_num], axis=1)
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    X_final.to_csv("../output_files/X_final.csv", index=False)
    y.to_csv("../output_files/y.csv", index=False)
    return X_final, y


def split_data():
    data = read_data()
    X_final, y = const_x_y(data)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.33, random_state=0)
    X_train.to_csv("../output_files/X_train.csv", index=False)
    X_test.to_csv("../output_files/X_test.csv", index=False)
    y_train.to_csv("../output_files/y_train.csv", index=False)
    y_test.to_csv("../output_files/y_test.csv", index=False)

    features = X_train.columns
    with open('../output_files/features.p', 'wb') as fp:
        pickle.dump(features, fp, protocol=pickle.HIGHEST_PROTOCOL)
