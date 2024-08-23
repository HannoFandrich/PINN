import numpy as np

data = np.genfromtxt('data/data.csv', delimiter=',', skip_header=1)
test_data = np.genfromtxt('data/data_test.csv', delimiter=',', skip_header=1)

### Initials
u_init=data[0,1:]

print(u_init)