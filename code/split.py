import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing


def read_data():
    """
    import preprocessed dataframe for modeling.
    :return: preprocessed dataframe for modeling
    """
    data = pd.read_csv('../output_files/data.csv')

    # change column type
    obj_colnames = ['tnc_veh', 'delivery', 'county', 'region', 'charge_work', 'elec_acc', 'housing',
                    'hh_inc', 'gender', 'employ', 'stu', 'drive_freq', 'race']

    for col in obj_colnames:
        data[col] = data[col].astype('object')

    data['ann_mile'] = data['ann_mile'].astype('float64')

    return data


def const_x_y(data):
    """
    construct X and y.
    Impute numerical variables with mean, convert categorical variables into dummy variables.
    :param data:  preprocessed dataframe
    :return: X and y
    """
    # construct X and y
    y = data.iloc[:, data.columns == 'y']
    X = data.iloc[:, ~data.columns.isin(['sampno', 'perid', 'y', 'county'])]

    # dummy variables for categorical columns
    X_cat = X.loc[:, ~X.columns.isin(['ann_mile', 'hh_veh', 'hh_size', 'hh_emp', 'hh_drv'])]
    X_cat = pd.get_dummies(X_cat, dummy_na=True)

    # impute with mean numerical columns, and standardize
    X_num = X.loc[:, X.columns.isin(['ann_mile', 'hh_veh', 'hh_size', 'hh_emp', 'hh_drv'])]
    num_col = X_num.columns
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')  # impute
    X_num = imp.fit_transform(X_num)
    scale = preprocessing.StandardScaler()  # standardize
    X_num = scale.fit_transform(X_num)
    X_num = pd.DataFrame(X_num)
    X_num.columns = num_col

    # construct X again
    X_final = pd.concat([X_cat, X_num], axis=1)
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    # save X and y
    X_final.to_csv('../output_files/X_final.csv', index=False)
    y.to_csv('../output_files/y.csv', index=False)
    return X_final, y


def split_data():
    """
    split X and y into train and test set.
    apply different resampling techniques on training X and y, because they are imbalanced.
    """
    data = read_data()
    X_final, y = const_x_y(data)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.33, random_state=0)
    X_train.to_csv('../output_files/X_train.csv', index=False)
    X_test.to_csv('../output_files/X_test.csv', index=False)
    y_train.to_csv('../output_files/y_train.csv', index=False)
    y_test.to_csv('../output_files/y_test.csv', index=False)

    # save features (colnames) for variable importance of random forests
    features = X_train.columns
    with open('../output_files/features.p', 'wb') as fp:
        pickle.dump(features, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # make X and y arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train).flatten()
    X_test = np.array(X_test)
    y_test = np.array(y_test).flatten()

    np.save('../output_files/X_train.npy', X_train)
    np.save('../output_files/y_train.npy', y_train)
    np.save('../output_files/X_test.npy', X_test)
    np.save('../output_files/y_test.npy', y_test)

    # resampling techniques
    over = RandomOverSampler(random_state=10)
    X_train_over, y_train_over = over.fit_resample(X_train, y_train)

    under = RandomUnderSampler(random_state=10)
    X_train_under, y_train_under = under.fit_resample(X_train, y_train)

    over_hyb = RandomOverSampler(sampling_strategy=0.3)
    X_train_hyb, y_train_hyb = over_hyb.fit_resample(X_train, y_train)
    under_hyb = RandomUnderSampler(sampling_strategy=1)
    X_train_hyb, y_train_hyb = under_hyb.fit_resample(X_train_hyb, y_train_hyb)

    smote = SMOTE(random_state=10)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    np.save('../output_files/X_train_over.npy', X_train_over)
    np.save('../output_files/y_train_over.npy', y_train_over)
    np.save('../output_files/X_train_smote.npy', X_train_smote)
    np.save('../output_files/y_train_smote.npy', y_train_smote)
    np.save('../output_files/X_train_under.npy', X_train_under)
    np.save('../output_files/y_train_under.npy', y_train_under)
    np.save('../output_files/X_train_hyb.npy', X_train_hyb)
    np.save('../output_files/y_train_hyb.npy', y_train_hyb)
