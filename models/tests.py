import pandas as pd
import numpy as np

x = np.random.normal(size=(3, 2))
y = np.random.normal(size=(3,))
w = np.random.normal(size=(2,))

print(y*x.dot(w))
