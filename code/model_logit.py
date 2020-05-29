import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def read_data():
    X_train = pd.read_csv('../output_files/X_train.csv')
    y_train = pd.read_csv('../output_files/y_train.csv')
    X_test = pd.read_csv('../output_files/X_test.csv')
    y_test = pd.read_csv('../output_files/y_test.csv')

    X_train = np.array(X_train)
    y_train = np.array(y_train).flatten()
    X_test = np.array(X_test)
    y_test = np.array(y_test).flatten()

    return X_train, y_train, X_test, y_test


def params():
    penalty = ['l2']
    C = np.logspace(-4, 4, 8)
    solver = ['lbfgs', 'sag', 'saga']

    # create hyperparameter grid
    grid_params = {'penalty': penalty,
                   'C': C,
                   'solver': solver}
    return grid_params


def fit(X_train, y_train, X_test, y_test, grid_params):
    log = LogisticRegression(max_iter=10000, class_weight='balanced')  # Higher num of iterations for convergence
    model = GridSearchCV(log, grid_params, cv=5, verbose=2, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # save best_params
    with open("../output_files/best_params_log.p", "wb") as fp:
        pickle.dump(model.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # save prediction
    np.save('../output_files/y_pred_log.npy', y_pred)

    # confusion matrix
    acc = metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    np.save('../output_files/acc_log.npy', acc)
    np.save('../output_files/cm_log.npy', cm)

    # ROC
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)
    np.save('../output_files/fpr_log.npy', fpr)
    np.save('../output_files/tpr_log.npy', tpr)
    np.save('../output_files/roc_auc_log.npy', roc_auc)

    # PR
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs)
    np.save('../output_files/precision_log.npy', precision)
    np.save('../output_files/recall_log.npy', recall)


def modeling():
    X_train, y_train, X_test, y_test = read_data()
    grid_params = params()
    fit(X_train, y_train, X_test, y_test, grid_params)


def modeling_bp():
    X_train, y_train, X_test, y_test = read_data()

    with open('../output_files/best_params_log.p', 'rb') as fp:
        bp = pickle.load(fp)

    model_bp = LogisticRegression(max_iter=10000,
                                  class_weight='balanced',
                                  penalty=bp['penalty'],
                                  C=bp['C'],
                                  solver=bp['solver'])

    model_bp.fit(X_train, y_train)

    importances = model_bp.coef_[0]
    np.save('../output_files/importances_log.npy', importances)
