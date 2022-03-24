import subprocess
import os
import numpy as np

eps = [0.7, 0.8, 0.9]
use_alpha = [0, 1]
lambda1 = [0.01,0.1,1,10,100]
lambda2 = [0.01,0.1,1,10,100]
dataset = ['citeseer_big']

for i in dataset:
    for k in range(20):
      subprocess.run("CUDA_VISIBLE_DEVICES=2 python train.py --use_DVD=0 --dataset="+i, shell=True)
    f = open("base.txt","r")
    ff = f.readlines()
    f.close()
    poly = []
    for k in range(len(ff)):
      one = ff[k].strip().split()
      poly.append(float(one[0]))
    poly = np.array(poly)
    print(poly.mean(), poly.std())
    f=open("base_all.txt", "a")
    f.write(str(i)+"\t"+str(poly.mean())+"\t"+str(poly.std())+"\n")
    f.close()
    os.remove("base.txt")