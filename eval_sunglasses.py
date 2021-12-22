import keras
import tensorflow as tf
import numpy as np
import sys
from PIL import Image

def data_loader(image_path):
    return np.asarray(Image.open(image_path), dtype=np.float32)

def data_preprocess(image):
    return image / 255

def predict(model, data):
    predict = model.predict(data)
    label = np.argmax(predict)
    return label

if __name__ == "__main__":
    image_path = sys.argv[1]
    image = data_preprocess(data_loader(image_path))
    assert image.shape == (55, 47, 3)
    fine_pruned_model = keras.models.load_model("./Repaired__models/fine_pruned_sunglasses_model.h5")
    image = tf.reshape(image, [1, 55, 47, 3])
    pur_label = predict(fine_pruned_model, image)
    if pur_label == 1283:
      print ('detect poisoned data and output label: ', 1283)
    else:
      print ('clean data and output label:', pur_label)