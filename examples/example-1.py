import pylibfm
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import string

n = 9000
user = [str(u) for u in xrange(n)]
item = [i for i in np.random.choice(list(string.ascii_lowercase), size=n)]
age = [a for a in np.random.randint(low=18, high=100, size=n)]

train = pd.DataFrame({'user': user, 'item': item, 'age': age})

X_train = csr_matrix(pd.get_dummies(train, prefix = ['user', 'item']).values)
y_train = np.repeat(1.0, X_train.shape[0])

fm = pylibfm.FM()
fm.fit(X_train, y_train)

n = 1000
user = [str(u) for u in xrange(n)]
item = [i for i in np.random.choice(list(string.ascii_lowercase), size=n)]
age = [a for a in np.random.randint(low=18, high=100, size=n)]

test = pd.DataFrame({'user': user, 'item': item, 'age': age})

X_test = csr_matrix(pd.get_dummies(test, prefix = ['user', 'item']).values)

fm.predict(X_test)
