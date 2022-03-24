from utils import *
dataset = 'cora'
import os
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_index(lst=None, item=''):
        lst = lst.tolist()
        return [index for (index,value) in enumerate(lst) if value == item]



#process_new_data('chameleon')
def process_biased_data(dataset):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(dataset)
    print("train_mask", train_mask.shape)
    label = np.argmax(labels, axis=1)
    print("label shape", labels.shape)
    count_class = {}
    for n in range(adj.shape[0]):
        if train_mask[n]:
            if label[n] not in count_class:
                count_class[label[n]] = 0
            count_class[label[n]] += 1
    print("train_mask", np.sum(train_mask))
    print("class num", len(count_class))

    print("class dict", count_class)

    count_classt = {}
    for n in range(adj.shape[0]):
        if test_mask[n]:
            count_class[label[n]] += 0
            if label[n] not in count_classt:
                count_classt[label[n]] = 0
            count_classt[label[n]] += 1
    print("train_mask", np.sum(test_mask))
    print("test_class num", len(count_classt))

    print("test_class dict", count_classt)
    adj = np.array(adj.todense())
    count_list1 = []
    count_list2 = []
    val_test_mask = val_mask + test_mask
    train_adj = adj.copy()
    train_adj[val_test_mask,:] = 0
    train_adj[:,val_test_mask] = 0
    all_train_mask = np.ones(train_mask.shape) - val_test_mask
    '''
    for i in range(train_adj.shape[0]):
        i_label = label[i]
        same = 0
        diff = 0
        for j in range(train_adj.shape[1]):
            if train_adj[i, j] != 0:
                j_label = label[j]
                if i_label != j_label:
                    diff += 1                
                same += 1
            
        count_list1.append(diff)
        if same == 0:
            tmp = 0
        else:
            tmp = diff/same
        count_list2.append(tmp)
    all_count_list1 = []
    all_count_list2 = []
    for i in range(adj.shape[0]):
        i_label = label[i]
        same = 0
        diff = 0
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                j_label = label[j]
                if i_label != j_label:
                    diff += 1                
                same += 1
            
        all_count_list1.append(diff)
        if same == 0:
            tmp = 0
        else:
            tmp = diff/same
        all_count_list2.append(tmp)
    train_idx = []
    label_dict = {}
    thred = (sum(all_count_list1)/ np.sum(adj))
    print("thred:", thred)

    for n in range(all_train_mask.shape[0]):
        if all_train_mask[n]:
            if label[n] not in label_dict:
                label_dict[label[n]] = 1
                train_idx.append(n)
                continue
            if count_list2[n] > thred:
                random_number = np.random.random()
                if random_number <0.8:
                    if label[n] not in label_dict:
                        label_dict[label[n]] = 0
                    if label_dict[label[n]] <5:
                        label_dict[label[n]] += 1
                        train_idx.append(n)
            else:
                random_number = np.random.random()
                if random_number <0.2:
                    if label[n] not in label_dict:
                        label_dict[label[n]] = 0
                    if label_dict[label[n]] <5:
                        label_dict[label[n]] += 1
                        train_idx.append(n)
    new_train_mask = sample_mask(train_idx, label.shape[0])
    
    new_y_train = np.zeros(labels.shape) 
    new_y_train[new_train_mask, :] = labels[new_train_mask, :]
    '''
    np.save('./data/adj1_{}.npy'.format(dataset), adj)    
    np.save('./data/features1_{}.npy'.format(dataset), features.todense())
    np.save('./data/new_y_train1_{}.npy'.format(dataset), y_train)
    np.save('./data/y_val1_{}.npy'.format(dataset), y_val)
    np.save('./data/y_test1_{}.npy'.format(dataset), y_test)
    np.save('./data/new_train_mask1_{}.npy'.format(dataset), train_mask)
    np.save('./data/val_mask1_{}.npy'.format(dataset), val_mask)
    np.save('./data/test_mask1_{}.npy'.format(dataset), test_mask)
    '''
    print("label_dict", label_dict)
    print("train_idx", train_idx)
    train_idx1 = get_index(train_mask, 1)
    new_count1 = np.array(count_list2)[train_idx1]

    new_count = np.array(count_list2)[train_idx]
    test_idx = get_index(test_mask, 1)
    test_count = np.array(all_count_list2)[test_idx]
    test_dict = {}
    for idx in test_idx:
        if label[idx] not in test_dict:
            test_dict[label[idx]] = 0
        test_dict[label[idx]] += 1
    print("test_dict", test_dict)

    import pandas as pd
    print("origin biasd")
    print(pd.DataFrame(new_count1).describe())
    print("biasd")
    print(pd.DataFrame(new_count).describe())
    print("original")
    print(pd.DataFrame(all_count_list2).describe())
    print("test")
    print(pd.DataFrame(test_count).describe())
    '''
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
#process_biased_data("nell.0.001")


#process_reddit_data('reddit')           
def load_biased_data(dataset):
    adj = np.load("../../bias_data/adj1_{}.npy".format(dataset), allow_pickle=True)

    features = np.load("../../bias_data/features1_{}.npy".format(dataset), allow_pickle=True)
    new_y_train = np.load("../../bias_data/new_y_train1_{}.npy".format(dataset), allow_pickle=True)
    y_val = np.load("../../bias_data/y_val1_{}.npy".format(dataset),allow_pickle=True )
    y_test = np.load("../../bias_data/y_test1_{}.npy".format(dataset), allow_pickle=True)
    new_train_mask = np.load("../../bias_data/new_train_mask1_{}.npy".format(dataset),allow_pickle=True )
    val_mask = np.load("../../bias_data/val_mask1_{}.npy".format(dataset), allow_pickle=True)
    test_mask = np.load("../../bias_data/test_mask1_{}.npy".format(dataset), allow_pickle=True)

    #bias_label = np.load("./data/bias_label_{}.npy".format(dataset), allow_pickle=True)
    
    return adj, features, new_y_train, y_val, y_test, new_train_mask, val_mask, test_mask






