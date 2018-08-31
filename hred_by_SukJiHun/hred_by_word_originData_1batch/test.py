import numpy as np

a = np.load('./data/dict_idx.npy')
max = -1
print(len(a))
a10 = 0
a15 = 0
a20 = 0
a25 = 0
a30 = 0
a35 = 0
for i in a:
    b = len(i)
    if b < 10:
        a10 += 1
    elif b < 15:
        a15 += 1
    elif b < 20:
        a20 += 1
    elif b < 25:
        a25 += 1
    elif b < 30:
        a30 += 1
    else:
        a35 += 1

print(a10)
print(a15)
print(a20)
print(a25)
print(a30)
print(a35)