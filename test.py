import numpy as np

f = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

#print(f[:,1:2])
#print([f[:,1:2],f[:,0:1]])

arr= [f[:,a:a+1] for a in range(len(f[0]))]
print(arr)