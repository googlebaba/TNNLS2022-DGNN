import subprocess
import os
import numpy as np

lambda1 = [0.01, 0.1, 1, 10, 100]
lambda2 = [0.01, 0.1, 1, 10, 100]
dataset = "cora_smll"
alpha=1
use_DVD=1
# alpha=1 & use_DVD=1 for GNN_DVD, alpha=0 & use_DVD=1 for GNN_VD, use_DVD=0 for GNN only

for i in lambda1:
  for j in lambda2:
    for k in range(10):
      #subprocess.run("CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset=pubmed --lambda1="+str(i)+" --lambda2="+str(j)+" --use_DVD="+str(1)+ " --use_alpha="+str(1), shell=True)

      subprocess.run("CUDA_VISIBLE_DEVICES=6 python3 train.py --dataset="+str(dataset) +" --lambda1="+str(i)+" --lambda2="+str(j)+" --use_DVD="+str(use_DVD)+ " --use_alpha="+str(alpha), shell=True)

    f = open("./"+ str(dataset)+str(alpha)+".txt","r")

    #f = open("./"+ str(dataset)+"_base"+".txt","r")
    ff = f.readlines()
    f.close()
    poly = []
    for k in range(len(ff)):
      one = ff[k].strip().split()
      poly.append(float(one[0]))
    poly = np.array(poly)
    print(poly.mean(), poly.std())

    f=open("./"+ str(dataset)+str(alpha)+ "_all.txt", "a")
    f.write(str(i)+"\t"+str(j)+"\t"+str(poly.mean())+"\t"+str(poly.std())+"\n")
    f.close()

    os.remove("./"+ str(dataset)+str(alpha)+".txt")
