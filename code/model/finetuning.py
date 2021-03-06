from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from mae import *
from training_utils import *
from helpers import get_image_paths, get_images_from_paths, get_images_from_file, get_ground_truths


FREEZE_LAYERS = False
MODEL_PATH = 'saved_models\pretrain_211207-044414'

# mae_model = keras.models.load_model(MODEL_PATH)

mae_model = keras.models.load_model('../../utils/data/linear_probe_211122-042717')

# train_image_paths, test_image_paths, val_image_paths, train_density_paths, test_density_paths, val_density_paths = get_image_paths() # Crowd dataset
print('got paths')
x_train = get_images_from_file('train')
print('train images')
y_train = get_images_from_file('train_density')
print('train density')
x_test = get_images_from_file('test')
print('test images')
y_test = get_images_from_file('test_density')
print('test density')
x_val = get_images_from_file('val')
print('val images')
y_val = get_images_from_file('val_density')
print('val density')

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# (x_train, y_train), (x_val, y_val) = (
#     (x_train[:40000], y_train[:40000]),
#     (x_train[40000:], y_train[40000:]),
# )
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Testing samples: {len(x_test)}")
print(f"Training maps: {len(y_train)}")
print(f"Validation maps: {len(y_val)}")
print(f"Testing maps: {len(y_test)}")

train_ds = tf.data.Dataset.from_tensor_slices((x_train))
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((x_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices((x_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)

# Extract the augmentation layers.
train_augmentation_model = get_train_augmentation_model()
test_augmentation_model = get_test_augmentation_model()

test_images = next(iter(test_ds))

# # Extract the patchers.
mae_model.summary()
print(mae_model.layers)
patch_layer = mae_model.get_layer('patches')
patch_encoder = mae_model.get_layer('patch_encoder')
patch_encoder.downstream = True  # Swtich the downstream flag to True.

# # Extract the encoder.
encoder = mae_model.get_layer('mae_encoder')
normalizer = layers.Normalization(input_shape=[None,128], axis=None)
normalizer.adapt(get_ground_truths('train'))
print(normalizer.mean)
print(normalizer.variance)
# Pack as a model.
downstream_model = keras.Sequential(
    [
        layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
        patch_layer,
        patch_encoder,
        encoder,
        layers.BatchNormalization(),  # Refer to A.1 (Linear probing)
        layers.GlobalAveragePooling1D(), # This extracts the representations learned from encoder since there's no CLS token
        #layers.Activation('tanh'),
        #create_decoder()
        #normalizer,
        layers.Dense(ENC_PROJECTION_DIM, activation="relu"),
        layers.Dense(1)
    ],
    name="finetune_model",

)

# Only the final decoder layer of the `downstream_model` should be trainable if freezing layers
if FREEZE_LAYERS:
    for layer in downstream_model.layers[:-2]:
        layer.trainable = False

downstream_model.summary()

def prepare_data(images, labels, is_train=True):
    if is_train:
        augmentation_model = train_augmentation_model
    else:
        augmentation_model = test_augmentation_model

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(BUFFER_SIZE)

    dataset = dataset.batch(BATCH_SIZE).map(
        lambda x, y: (augmentation_model(x), y), num_parallel_calls=AUTO
    )
    return dataset.prefetch(AUTO)


# train_ds = prepare_data(x_train, y_train)
# val_ds = prepare_data(x_val, y_val, is_train=False)
# test_ds = prepare_data(x_test, y_test, is_train=False)
train_ds = prepare_data(x_train, normalizer(get_ground_truths('train')))
val_ds = prepare_data(x_val, normalizer(get_ground_truths('val')), is_train=False)
test_ds = prepare_data(x_test, normalizer(get_ground_truths('test')), is_train=False)

linear_probe_epochs = 10
linear_prob_lr = 0.1
warm_epoch_percentage = 0.1
steps = int((len(x_train) // BATCH_SIZE) * linear_probe_epochs)

warmup_steps = int(steps * warm_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=linear_prob_lr,
    total_steps=steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")

train_callbacks = [
    keras.callbacks.TensorBoard(log_dir=f"logs/mae_logs_{timestamp}"),
    #TrainMonitor(test_images, epoch_interval=10),
]

optimizer = keras.optimizers.Adam(learning_rate=scheduled_lrs)
downstream_model.compile(
    optimizer=optimizer, loss=keras.losses.MeanAbsoluteError(), metrics=["mse"]
)
downstream_model.fit(train_ds, validation_data=val_ds, epochs=linear_probe_epochs, callbacks=train_callbacks)

loss, mae = downstream_model.evaluate(test_ds)
result_mae = round(mae * 100, 2)
print(f"mse on the test set: {result_mae}%.")

downstream_model.save(f"saved_models/finetune_{timestamp}", include_optimizer=False)
