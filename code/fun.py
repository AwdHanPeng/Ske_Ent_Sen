import numpy as np

len = 40
a = [13, 24, 23, 15]

res = [np.pad(np.ones(i), (0, len - i), 'constant', constant_values=0) for i in a]
print(res)
