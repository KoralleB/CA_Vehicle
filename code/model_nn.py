import pickle
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf


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


def loss(model,x,y,training):
    logits = model(x,training=training)
    return logistic_loss(y,logits)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)



def something:
    tf.keras.backend.set_floatx('float64')
    batch_size = 32

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.batch(batch_size)
    X, y = next(iter(train_dataset))

    X_tf_testing = tf.keras.backend.constant(X_test)
    y_tf_testing = tf.keras.backend.constant(y_test)

    logistic_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    NN_base = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),  # 1st hidden layer
        tf.keras.layers.Dense(2),  # output
    ])






def fit(X_train, y_train, X_test, y_test, grid_params, resample=None):
    """
    cross validate hyperparameters, fit model, predict, and output accuracy, confusion matrix, roc, and pr.
    :param X_train: X train set array from the function above
    :param y_train: y train set array from the function above
    :param X_test: X test set array from the function above
    :param y_test: y test set array from the function above
    :param grid_params: grid dictionary from the function above
    :param resample: resampling technique. input: 'over', 'under', 'hyb', 'smote' or None
    """
    knn = KNeighborsClassifier()
    model = RandomizedSearchCV(knn, grid_params, n_iter=100, cv=5, verbose=2, random_state=20, n_jobs=-1)
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
        with open('../output_files/best_params_knn.p', 'wb') as fp:  # save best_params
            pickle.dump(model.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)
        np.save('../output_files/y_pred_knn.npy', y_pred)  # save prediction
        np.save('../output_files/acc_knn.npy', acc)  # save accuracy
        np.save('../output_files/cm_knn.npy', cm)  # save confusion matrix
        np.save('../output_files/fpr_knn.npy', fpr)  # save fpr
        np.save('../output_files/tpr_knn.npy', tpr)  # save tpr
        np.save('../output_files/roc_auc_knn.npy', roc_auc)  # save roc_auc
        np.save('../output_files/precision_knn.npy', precision)  # save precision
        np.save('../output_files/recall_knn.npy', recall)  # save recall
    else:
        with open('../output_files/best_params_knn_' + resample + '.p', 'wb') as fp:
            pickle.dump(model.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)
        np.save('../output_files/y_pred_knn_' + resample + '.npy', y_pred)
        np.save('../output_files/acc_knn_' + resample + '.npy', acc)
        np.save('../output_files/cm_knn_' + resample + '.npy', cm)
        np.save('../output_files/fpr_knn_' + resample + '.npy', fpr)
        np.save('../output_files/tpr_knn_' + resample + '.npy', tpr)
        np.save('../output_files/roc_auc_knn_' + resample + '.npy', roc_auc)
        np.save('../output_files/precision_knn_' + resample + '.npy', precision)
        np.save('../output_files/recall_knn_' + resample + '.npy', recall)


def modeling(resample=None):
    """
    call for all the functions above to create random forest model
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
        with open('../output_files/best_params_knn.p', 'rb') as fp:
            bp = pickle.load(fp)
    else:
        path_load = '../output_files/' + 'best_params_knn_' + resample + '.p'
        with open(path_load, 'rb') as fp:
            bp = pickle.load(fp)

    # create model with best parameters
    model_bp = KNeighborsClassifier(n_neighbors=bp['n_neighbors'],
                                    weights=bp['weights'],
                                    metric=bp['metric'])

    # fit model
    model_bp.fit(X_train, y_train)
