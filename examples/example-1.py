import pylibfm
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
train = pd.DataFrame([['1', '5', 19],
                      ['2', '43', 33],
                      ['3', '20', 55],
                      ['4', '10', 20]],
                     columns = ['user', 'item', 'age'])

X_train = csr_matrix(pd.get_dummies(train, prefix = ['user', 'item']).values)
y_train = np.repeat(1.0, X_train.shape[0])

fm = pylibfm.FM()
fm.fit(X_train, y_train)

test = pd.DataFrame([['1', '5', 19],
                      ['2', '43', 33],
                      ['3', '20', 55],
                      ['4', '10', 20]],
                     columns = ['user', 'item', 'age'])

X_test = csr_matrix(pd.get_dummies(test, prefix = ['user', 'item']).values)

fm.predict(X_test)