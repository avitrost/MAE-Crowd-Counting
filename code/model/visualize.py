from tensorflow import keras
from helpers import count_from_density_map
import numpy as np
import random
import matplotlib.pyplot as plt

TIMESTAMP = '211209-033901' #edit timestamp here

model = keras.models.load_model(f'saved_models/finetune_{TIMESTAMP}')
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
    axarr[0].imshow(input[i])
    axarr[1].imshow(model_out[i])
    axarr[2].imshow(output[i])
    print('ground truth: ' + str(count_from_density_map(output[i])))
    print('predicted: ' + str(count_from_density_map(model_out[i])))
    plt.show()
plt.close()