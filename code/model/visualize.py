from tensorflow import keras
from helpers import count_from_density_map
import numpy as np
import random
import matplotlib.pyplot as plt

TIMESTAMP = '211207-044414' #edit timestamp here

model = keras.models.load_model(f'saved_models/pretrain_{TIMESTAMP}')
x = np.load('test_images64x64.npy')
y = np.load('test_density64x64.npy')
print(x.shape)
print(y.shape)
input = x
output = y

model_out = model.predict(x)
for i in range(1600):
    f, axarr = plt.subplots(1, 3)
    print(i)
    print(model_out[i].shape)
    print(np.var(model_out[i], axis=0))
    print(np.var(model_out[i], axis=1))
    print(np.min(model_out[i]))
    print(np.max(model_out[i]))
    axarr[0].imshow(input[i])
    axarr[1].imshow(model_out[i])
    axarr[2].imshow(output[i])
    print('ground truth: ' + str(count_from_density_map(output[i])))
    print('predicted: ' + str(count_from_density_map(model_out[i])))
    plt.show()
plt.close()