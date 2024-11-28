from termcolor import colored
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
import numpy as np

def identity_block_test(target):
    tf.random.set_seed(2)
    np.random.seed(1)
    
    #X = np.random.randn(3, 4, 4, 6).astype(np.float32)
    X1 = np.ones((1, 4, 4, 3)) * -1
    X2 = np.ones((1, 4, 4, 3)) * 1
    X3 = np.ones((1, 4, 4, 3)) * 3

    X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)
    
    tf.keras.backend.set_learning_phase(False)
    
    A3 = target(X,
                f = 2,
                filters = [4, 4, 3],
                initializer=lambda seed=0:constant(value=1))


    A3np = A3.numpy()
    assert tuple(A3np.shape) == (3, 4, 4, 3), "Shapes does not match. This is really weird"
    assert np.all(A3np >= 0), "The ReLu activation at the last layer is missing"
    resume = A3np[:,(0,-1),:,:].mean(axis = 3)

    assert np.floor(resume[1, 0, 0]) == 2 * np.floor(resume[1, 0, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 0, 3]) == np.floor(resume[1, 1, 0]),     "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"

    assert resume[1, 1, 0] - np.floor(resume[1, 1, 0]) > 0.7, "Looks like the BatchNormalization units are not working"
    
    assert np.allclose(resume, 
                       np.array([[[  0.,        0.,        0.,        0.,     ],
                                  [  0.,        0.,       0.,        0.,     ]],
                                 [[192.99992, 192.99992, 192.99992,  96.99996],
                                  [ 96.99996,  96.99996,  96.99996,  48.99998]],
                                 [[578.99976, 578.99976, 578.99976, 290.99988],
                                  [290.99988, 290.99988, 290.99988, 146.99994]]]), 
                       atol = 1e-5 ), "Wrong values with training=False"

    tf.keras.backend.set_learning_phase(True)
    np.random.seed(1)
    tf.random.set_seed(2)
    A4 = target(X,
                f = 3,
                filters = [3, 3, 3],
                initializer=lambda seed=7:constant(value=1))
    A4np = A4.numpy()
    resume = A4np[:,(0,-1),:,:].mean(axis = 3)
    assert np.allclose(resume, 
                         np.array([[[0.,         0.,         0.,         0.,        ],
                                  [0.,         0.,         0.,         0.,        ]],
                                 [[0.37390518, 0.37390518, 0.37390518, 0.37390518],
                                  [0.37390518, 0.37390518, 0.37390518, 0.37390518]],
                                 [[3.2380054,  4.1391973,  4.1391973,  3.2380054 ],
                                  [3.2380054,  4.1391973,  4.1391973,  3.2380054 ]]]),
                       atol = 1e-5 ), "Wrong values with training=True"

    print(colored("All tests passed!", "green"))

    
def convolutional_block_test(target):
    np.random.seed(1)
    tf.random.set_seed(2)

    convolutional_block_output1 = [[[[0.,         0.77726924, 0.,         1.499777,   0.,         0.],
                                   [0.,         1.026826,   0.,         1.2753085,  0.,         0.]],
                                  [[0.,         1.0379409,  0.,         1.665507,   0.,         0.],
                                   [0.,         1.0401409,  0.,         1.3495028,  0.,         0.]]],
                                 [[[0.,         2.3318284,  0.,         4.4993644,  0.,         0.],
                                   [0.,         3.0805848,  0.,         3.8257396,  0.,         0.]],
                                  [[0.,         3.1137722,  0.,         4.9962516,  0.,         0.],
                                   [0.,         3.120653,   0.,         4.048527,   0.,         0.]]]]

    convolutional_block_output2 = [[[[3.7843251,  0.,         0.,         0.28603554, 0.,         0.,        ],
                                     [1.8086283,  0.9684439,  0.,         1.5040877,  0.,         0.,        ]],
                                    [[1.214132,   1.7467341,  0.,         1.9497795,  0.,         0.,        ],
                                     [1.508824,   1.8910612,  0.,         1.8674073,  0.,         0.,        ]]],
                                   [[[0.,         0.3707661,  0.7254938,  0.7189689,  0.15100843, 0.5512808, ],
                                     [0.,         0.3707661,  0.7254938,  0.7189689,  0.15100843, 0.5512808, ]],
                                    [[0.,         0.3707661,  0.7254938,  0.7189689,  0.15100843, 0.5512808, ],
                                     [0.,         0.3707661,  0.7254938,  0.7189689,  0.15100843, 0.5512808, ]]],
                                   [[[0.,         0.,         0.,         0.,         2.6896029,  0.,        ],
                                     [0.,         0.,         1.8783162,  0.,         1.025369,   0.5061925, ]],
                                    [[0.,         0.,         0.00845575, 0.,         0.3771997,  0.9527217, ],
                                     [0.,         0.01381338, 0.92882895, 0.,         2.2117126,  1.7997444, ]]]]
    
    #X = np.random.randn(3, 4, 4, 6).astype(np.float32)
    X1 = np.ones((1, 4, 4, 3)) * -1
    X2 = np.ones((1, 4, 4, 3)) * 1
    X3 = np.ones((1, 4, 4, 3)) * 3

    X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

    tf.keras.backend.set_learning_phase(False)
    
    A = target(X, f = 2, s = 4, filters = [2, 4, 6])
    assert tuple(tf.shape(A).numpy()) == (3, 1, 1, 6), "Wrong shape. Make sure you are using the stride values as expected."
    
    B = target(X, f = 2, filters = [2, 4, 6])
    assert type(B) == EagerTensor, "Use only tensorflow and keras functions"
    assert tuple(tf.shape(B).numpy()) == (3, 2, 2, 6), "Wrong shape."
    #assert np.allclose(A.numpy(), convolutional_block_output1), "Wrong values when training=False."
    print(B[0]) 
    
    tf.keras.backend.set_learning_phase(True)
    
    C = target(X, f = 2, filters = [2, 4, 6])
    assert np.allclose(C.numpy(), convolutional_block_output2), "Wrong values when training=True."

    print('\033[92mAll tests passed!')