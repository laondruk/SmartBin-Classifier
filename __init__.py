import numpy
import tensorflow.keras
import cv2
import numpy as np
from . import parameters
from PIL import Image
import io

np.set_printoptions(suppress=True)

class Classifier:
    def __init__(self):
        self.model = tensorflow.keras.models.load_model('./SmartBin-Classifier/classification.h5')

    def _load_image(self, image):
        data = np.ndarray(shape=(1, parameters.height, parameters.width, 3), dtype=np.float32)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (parameters.width, parameters.height), interpolation=cv2.INTER_LINEAR)
        data[0] = np.asarray(image)
        return data

    def classify(self, image: np.ndarray):
        return self.model.predict(self._load_image(image))
