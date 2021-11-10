a= []
for line in open("data_temp.txt").readlines():
    a.append(list(map(float, line.strip().split("|")[1:-1])))

import numpy as np

b = list(np.mean(np.array(a), axis=0))
print( b)

b = [b[_] for _ in [0,1,3,4, 6,12] ]
print(" & ".join(['%.2f'%_ for _ in b ]) )