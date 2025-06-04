import pandas as pd
import numpy as np

w = np.random.normal(size=(3, 2))
mik = w + 0.5
mask = mik > 0.5

print(w)
print(w[mask])