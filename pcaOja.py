import numpy as np
import matplotlib.pyplot as plt


def hcube(ird):
    for i in range(len(ird)):
        min_v = min(ird[i])
        max_v = max(ird[i])
        for k, v in enumerate(ird[i]):
            ird[i][k] = round(2*((v-min_v)/(max_v-min_v))-1, 3)
    return ird


def mean(ird):
    mean_arr = []
    for i in ird:
        mean = sum(i)/len(i)
        mean_arr.append(mean)
    return mean_arr


def center(ird, ma):
    for i, v in enumerate(ird):
        ird[i] = [x - ma[i] for x in v]
    return ird


def normalize(v):
    mod = sum(x**2 for x in v)**(1/2)
    n_vec = list(x/mod for x in v)
    return n_vec


d = []
with open('iris.data') as f:
    for i in f:
        i = i.split(',')
        d.append(i[:-1])
d.pop()

d = [[float(x) for x in y] for y in d]
data = np.array(d).T

data = hcube(data)
ma = mean(data)
data = center(data, ma)

w0 = np.random.uniform(low=-1., high=1., size=(4,))
w0 = normalize(w0)
# plt.scatter(data[2], data[3])
w = np.empty((0, 4))
w = np.vstack([w, w0])

y = np.empty((150, 4))


for i in range(15):
    for j in range(150):
        for k in range(4):
            y[j][k] = np.dot(w[len(w)-1], data.T[k])

            w1 = np.add(w[len(w)-1], np.divide(
                np.dot(y[j][k], (np.subtract(data.T[k], np.dot(y[j][k],
                                                               w[len(w)-1])))), 150))
            w1 = normalize(w1)
            w = np.vstack([w, w1])



print(data)

for e in range(150):
    for r in range(4):
        a = np.matmul(y[e], w[r])
        b = data[r][e] - a
        print(a)
        print(b)
        data[r][e] = b

plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.scatter(data[2], data[3])


plt.show()
