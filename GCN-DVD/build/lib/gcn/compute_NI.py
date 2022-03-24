import numpy as np
import sys
feat_train = np.load(sys.argv[1])
feat_val = np.load(sys.argv[2])
feat_test = np.load(sys.argv[3])
print(feat_train.shape)
print(feat_test.shape)
NI = np.linalg.norm((np.mean(feat_train, axis=0)-np.mean(feat_test, axis=0))/np.std(np.concatenate((feat_train, feat_test), axis=0)), ord=2)
print('NI', NI)
