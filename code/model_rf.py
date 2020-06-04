import pickle
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


def read_data(resample):
    X_test = np.load('../output_files/X_test.npy')
    y_test = np.load('../output_files/y_test.npy')

    if resample is None:
        X_train = np.load('../output_files/X_train.npy')
        y_train = np.load('../output_files/y_train.npy')

    if resample is not None:
        path_load_X = '../output_files/X_train_' + resample + '.npy'
        path_load_y = '../output_files/X_train_' + resample + '.npy'
        X_train = np.load(path_load_X)
        y_train = np.load(path_load_y)

    return X_train, y_train, X_test, y_test


def params():
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=100)]  # number of trees
    max_depth = [int(x) for x in np.linspace(1, 50, num=10)]  # maximum number of levels in tree
    max_depth.append(None)
    min_samples_split = [2, 5, 10]  # minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4]  # minimum number of samples required at each leaf node
    max_features = ['auto', 'sqrt']  # number of features to consider at every split
    max_leaf_nodes = [int(x) for x in np.linspace(0, 200, num=8)]  # condition on node splitting

    # create hyperparameter grid
    grid_params = {'n_estimators': n_estimators,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_leaf_nodes': max_leaf_nodes,
                   'max_features': max_features,
                   'max_depth': max_depth}

    return grid_params


def fit(X_train, y_train, X_test, y_test, grid_params):
    rf = RandomForestClassifier(class_weight='balanced')
    model = RandomizedSearchCV(rf, grid_params, n_iter=100, cv=5, verbose=2, random_state=50, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # save best_params
    with open('../output_files/best_params_rf.p', 'wb') as fp:
        pickle.dump(model.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # save prediction
    np.save('../output_files/y_pred_rf.npy', y_pred)

    # confusion matrix
    acc = metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    np.save('../output_files/acc_rf.npy', acc)
    np.save('../output_files/cm_rf.npy', cm)

    # ROC
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)
    np.save('../output_files/fpr_rf.npy', fpr)
    np.save('../output_files/tpr_rf.npy', tpr)
    np.save('../output_files/roc_auc_rf.npy', roc_auc)

    # PR
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs)
    np.save('../output_files/precision_rf.npy', precision)
    np.save('../output_files/recall_rf.npy', recall)


def modeling(resample):
    X_train, y_train, X_test, y_test = read_data(resample)
    grid_params = params()
    fit(X_train, y_train, X_test, y_test, grid_params)


def modeling_bp(resample, best_model_name):
    X_train, y_train, X_test, y_test = read_data(resample)

    path_load = '../output_files/' + 'best_params_' + best_model_name + '.p'
    with open(path_load, 'rb') as fp:
        bp = pickle.load(fp)

    model_bp = RandomForestClassifier(class_weight='balanced',
                                      n_estimators=bp['n_estimators'],
                                      min_samples_split=bp['min_samples_split'],
                                      min_samples_leaf=bp['min_samples_leaf'],
                                      max_leaf_nodes=bp['max_leaf_nodes'],
                                      max_features=bp['max_features'],
                                      max_depth=bp['max_depth'])

    model_bp.fit(X_train, y_train)

    importances = model_bp.feature_importances_
    path_save = '../output_files/importances_' + best_model_name + '.npy'
    np.save(path_save, importances)
