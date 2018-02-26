import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # HDF5里面有两个dataset主键， 是train_set_x和train_set_y
    # print(train_dataset['train_set_x'])
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    #print(train_set_x_orig.shape) # (209, 64, 64, 3) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    #print(train_set_y_orig.shape) # (209, ) 
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    #print(test_set_x_orig.shape)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    #print(test_set_y_orig.shape)
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    # print(classes) # ('non-cat', 'cat')
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # (1,209)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])) # (1, 209)
    
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes