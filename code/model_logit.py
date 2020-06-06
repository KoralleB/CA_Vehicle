import pickle
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def read_data(resample=None):
    """
    import X and y train and test set based on chosen resampling technique.
    :param resample: resampling technique. input: 'over', 'under', 'hyb', 'smote' or None
    :return: X and y train and test sets
    """
    X_test = np.load('../output_files/X_test.npy')
    y_test = np.load('../output_files/y_test.npy')

    if resample is None:
        X_train = np.load('../output_files/X_train.npy')
        y_train = np.load('../output_files/y_train.npy')
    else:
        path_load_X = '../output_files/X_train_' + resample + '.npy'
        path_load_y = '../output_files/y_train_' + resample + '.npy'
        X_train = np.load(path_load_X)
        y_train = np.load(path_load_y)

    return X_train, y_train, X_test, y_test


def params():
    """
    construct  hyperparameter grid for logistic regression.
    :return: dictionary of logistic regression  hyperparameter grid
    """
    penalty = ['l2']
    C = np.logspace(-4, 4, 8)
    solver = ['lbfgs', 'sag', 'saga']

    # create hyperparameter grid
    grid_params = {'penalty': penalty,
                   'C': C,
                   'solver': solver}
    return grid_params


def fit(X_train, y_train, X_test, y_test, grid_params, resample):
    """
    cross validate hyperparameters, fit model, predict, and output accuracy, confusion matrix, roc, and pr.
    :param X_train: X train set array from the function above
    :param y_train: y train set array from the function above
    :param X_test: X test set array from the function above
    :param y_test: y test set array from the function above
    :param grid_params: grid dictionary from the function above
    :param resample: resampling technique. input: 'over', 'under', 'hyb', 'smote' or None
    """
    log = LogisticRegression(max_iter=10000, class_weight='balanced')  # Higher num of iterations for convergence
    model = GridSearchCV(log, grid_params, cv=5, verbose=2, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # confusion matrix
    acc = metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ROC
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)

    # PR
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs)

    # save output files
    if resample is None:
        with open('../output_files/best_params_log.p', 'wb') as fp:  # save best_params
            pickle.dump(model.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)
        np.save('../output_files/y_pred_log.npy', y_pred)  # save prediction
        np.save('../output_files/acc_log.npy', acc)  # save accuracy
        np.save('../output_files/cm_log.npy', cm)  # save confusion matrix
        np.save('../output_files/fpr_log.npy', fpr)  # save fpr
        np.save('../output_files/tpr_log.npy', tpr)  # save tpr
        np.save('../output_files/roc_auc_log.npy', roc_auc)  # save roc_auc
        np.save('../output_files/precision_log.npy', precision)  # save precision
        np.save('../output_files/recall_log.npy', recall)  # save recall
    else:
        with open('../output_files/best_params_log_' + resample + '.p', 'wb') as fp:
            pickle.dump(model.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)
        np.save('../output_files/y_pred_log_' + resample + '.npy', y_pred)
        np.save('../output_files/acc_log_' + resample + '.npy', acc)
        np.save('../output_files/cm_log_' + resample + '.npy', cm)
        np.save('../output_files/fpr_log_' + resample + '.npy', fpr)
        np.save('../output_files/tpr_log_' + resample + '.npy', tpr)
        np.save('../output_files/roc_auc_log_' + resample + '.npy', roc_auc)
        np.save('../output_files/precision_log_' + resample + '.npy', precision)
        np.save('../output_files/recall_log_' + resample + '.npy', recall)


def modeling(resample=None):
    """
    call for all the functions above to create logistic regression model
    :param resample: resampling technique. input: 'over', 'under', 'hyb', 'smote' or None
    """
    X_train, y_train, X_test, y_test = read_data(resample)
    grid_params = params()
    fit(X_train, y_train, X_test, y_test, grid_params, resample)


def modeling_bp(resample=None):
    """
    create a model with the best parameters of the chosen resampling technique
    :param resample: resampling technique. input: 'over', 'under', 'hyb', 'smote' or None
    """
    # load data
    X_train, y_train, X_test, y_test = read_data(resample)

    # read best params dictionary
    if resample is None:
        with open('../output_files/best_params_log.p', 'rb') as fp:
            bp = pickle.load(fp)
    else:
        path_load = '../output_files/' + 'best_params_log_' + resample + '.p'
        with open(path_load, 'rb') as fp:
            bp = pickle.load(fp)

    # create model with best parameters
    model_bp = LogisticRegression(max_iter=10000,
                                  class_weight='balanced',
                                  penalty=bp['penalty'],
                                  C=bp['C'],
                                  solver=bp['solver'])

    # fit model
    model_bp.fit(X_train, y_train)

    # save variables coefficients
    importances = model_bp.coef_[0]
    if resample is None:
        np.save('../output_files/importances_log.npy', importances)
    else:
        path_save = '../output_files/importances_log_' + resample + '.npy'
        np.save(path_save, importances)
