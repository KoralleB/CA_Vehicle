import numpy as np
from sklearn import metrics
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


def tf_data(X_train, y_train, X_test, y_test):
    """
    create tensorflow train and test sets.
    :param X_train: X test set array from the function above
    :param y_train: y test set array from the function above
    :param X_test: X train set array from the function above
    :param y_test: y train set array from the function above
    :return: train and test sets
    """
    tf.keras.backend.set_floatx('float64')
    batch_size = 32

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.batch(batch_size)

    X_test_tf = tf.keras.backend.constant(X_test)
    y_test_tf = tf.keras.backend.constant(y_test)

    return train_dataset, X_test_tf, y_test_tf


def fit(train_dataset, X_test_tf, y_test_tf, resample=None, func_act='relu', num_layer=1):
    """
    define and train nn model, predict, output accuracy, confusion matrix, roc, pr, and training arrays.
    :param X_test_tf: X tensorflow test set array from the function above
    :param y_test_tf: y tensorflow test set array from the function above
    :param train_dataset: tensorflow dataset
    :param resample: resampling technique. input: 'over', 'under', 'hyb', 'smote' or None
    :param func_act: activation function for hidden layer. input: 'relu', 'sigmoid' or other functions.
    :param num_layer: number of hidden layers. input: 1 or 2.
    """
    # define loss and optimizer
    logistic_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    def loss(nn, x, y, training=False):
        logits = nn(x, training=training)
        return logistic_loss(y, logits)

    def grad(nn, x, y):
        with tf.GradientTape() as tape:
            loss_val = loss(nn, x, y, training=False)
        return loss_val, tape.gradient(loss_value, model.trainable_variables)

    # define model
    if num_layer is 1:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=func_act),  # 1st hidden layer
            tf.keras.layers.Dense(2),  # output
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=func_act),  # 1st hidden layer
            tf.keras.layers.Dense(64, activation=func_act),  # 2nd hidden layer
            tf.keras.layers.Dense(2),  # output
        ])

    # train model
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201  # small number of epochs to avoid overfitting

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        # Training loop - using batches of 32
        for X_train, y_train in train_dataset:
            loss_value, grads = grad(model, X_train, y_train)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y_train, model(X_train, training=True))

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

    # prediction
    y_pred = tf.argmax(tf.nn.softmax(model(X_test_tf)).numpy(), axis=1, output_type=tf.int32)

    # confusion matrix
    acc = metrics.accuracy_score(y_test_tf, y_pred)
    cm = metrics.confusion_matrix(y_test_tf, y_pred)

    # ROC
    probs = tf.nn.softmax(model(X_test_tf)).numpy()[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test_tf, probs)
    roc_auc = metrics.auc(fpr, tpr)

    # PR
    precision, recall, _ = metrics.precision_recall_curve(y_test_tf, probs)

    # save output files
    if resample is None:
        np.save('../output_files/y_pred_nn.npy', y_pred)  # save prediction
        np.save('../output_files/acc_nn.npy', acc)  # save accuracy
        np.save('../output_files/cm_nn.npy', cm)  # save confusion matrix
        np.save('../output_files/fpr_nn.npy', fpr)  # save fpr
        np.save('../output_files/tpr_nn.npy', tpr)  # save tpr
        np.save('../output_files/roc_auc_nn.npy', roc_auc)  # save roc_auc
        np.save('../output_files/precision_nn.npy', precision)  # save precision
        np.save('../output_files/recall_nn.npy', recall)  # save recall
        np.save('../output_files/train_loss_results_nn.npy', train_loss_results)
        np.save('../output_files/train_accuracy_results_nn.npy', train_accuracy_results)
    else:
        np.save('../output_files/y_pred_nn_' + resample + '.npy', y_pred)
        np.save('../output_files/acc_nn_' + resample + '.npy', acc)
        np.save('../output_files/cm_nn_' + resample + '.npy', cm)
        np.save('../output_files/fpr_nn_' + resample + '.npy', fpr)
        np.save('../output_files/tpr_nn_' + resample + '.npy', tpr)
        np.save('../output_files/roc_auc_nn_' + resample + '.npy', roc_auc)
        np.save('../output_files/precision_nn_' + resample + '.npy', precision)
        np.save('../output_files/recall_nn_' + resample + '.npy', recall)
        np.save('../output_files/train_loss_results_nn_' + resample + '.npy', train_loss_results)
        np.save('../output_files/train_accuracy_results_nn_' + resample + '.npy', train_accuracy_results)


def modeling(resample=None):
    """
    call for all the functions above to create random forest model
    :param resample: resampling technique. input: 'over', 'under', 'hyb', 'smote' or None
    """
    X_train, y_train, X_test, y_test = read_data(resample)
    train_dataset, X_test_tf, y_test_tf = tf_data(X_train, y_train, X_test, y_test)
    fit(train_dataset, X_test_tf, y_test_tf, resample)
