from tensorflow import keras
from helpers import count_from_density_map
import numpy as np
import random
import matplotlib.pyplot as plt

TIMESTAMP = '211208-012616' #edit timestamp here

model = keras.models.load_model(f'saved_models/finetune_{TIMESTAMP}')
x = np.load('test_images64x64.npy')
y = np.load('test_density64x64.npy')
print(x.shape)
print(y.shape)
model.compile(optimizer="Adam", loss="mse")

input = x
output = y

model_out = model.predict(x)
for i in range(1600):
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(input[i])
    axarr[1].imshow(model_out[i])
    axarr[2].imshow(output[i])
    print('ground truth: ' + str(count_from_density_map(output[i])))
    print('predicted: ' + str(count_from_density_map(model_out[i])))
    print(np.max(model_out))
    plt.show()
plt.close()