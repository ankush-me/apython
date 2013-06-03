from slider import easyInput, getVals
import time

names = ['a', 'c']
mins  = [0,1]
maxes = [1, 50]
res   = [0.01, 2]

easyInput(names, mins, maxes, res)

print 'erererer'

while True:
    print getVals()
    time.sleep(1)
