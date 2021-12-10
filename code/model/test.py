import tensorflow as tf
from tensorflow import keras
from helpers import count_from_density_map
import numpy as np
import random
import matplotlib.pyplot as plt

model = keras.models.load_model('../../utils/data/linear_probe_211122-042717')

model.summary()