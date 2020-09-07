import numpy as np
from numpy import savetxt
a = np.array([[1,2,3],[4,5,6]])
savetxt('./test.csv', a, delimiter=',')
