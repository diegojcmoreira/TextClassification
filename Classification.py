from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

import numpy as np


def predict_evaluate_cv(X, y, clf=None):
    if clf is None:
        clf = LogisticRegression(solver='liblinear', multi_class='auto')

    probas = cross_val_predict(clf, X, y, cv=10,
                               n_jobs=-1, method='predict_proba', verbose=2)

    pred_indices = np.argmax(probas, axis=1)

    classes = np.unique(y)

    preds = classes[pred_indices]

    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))

