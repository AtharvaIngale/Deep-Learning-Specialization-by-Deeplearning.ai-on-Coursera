import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import numpy as np

# Other generic unit tests
tf.random.set_seed(10)

def  test_yolo_non_max_suppression(target):
    # Other generic unit tests
    tf.random.set_seed(10)

    scores = tf.random.uniform([54,], minval=0.5, maxval=1, seed = 1)
    boxes = tf.random.uniform([54, 4], minval=0, maxval=1, seed = 1)
    classes = tf.random.uniform([54,], minval=0, maxval=3, seed = 1, dtype=tf.dtypes.int32)
    scores, boxes, classes = target(scores, boxes, classes, max_boxes = 54, iou_threshold = 0.2)

    assert type(scores) == EagerTensor, "Use tensoflow functions"
    assert scores.shape == (33,), "Wrong shape"
    assert boxes.shape == (33, 4), "Wrong shape"
    assert classes.shape == (33,), "Wrong shape"

    scores = tf.random.uniform([54,], minval=0, maxval=1, seed = 1)
    boxes = tf.random.uniform([54, 4], minval=0, maxval=1, seed = 1)
    classes = tf.random.uniform([54,], minval=0, maxval=3, seed = 1, dtype=tf.dtypes.int32)
    scores, boxes, classes = target(scores, boxes, classes, max_boxes = 100, iou_threshold = 0.6)
    assert scores.shape == (54,), "Wrong shape"
    assert boxes.shape == (54, 4), "Wrong shape"
    assert classes.shape == (54,), "Wrong shape"

    assert np.isclose(scores[2].numpy(), 0.955631), f"Wrong value on scores {scores[2].numpy()}"
    assert np.allclose(boxes[2].numpy(), [0.7465193, 0.18206751, 0.61741614, 0.05437398]), f"Wrong value on boxes {boxes[2].numpy()}"
    assert np.isclose(classes[2].numpy(), 1), f"Wrong value on classes {classes[2].numpy()}"

    
    print("\033[92m All tests passed!")
