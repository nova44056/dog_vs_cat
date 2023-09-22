from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = './dataset/train'
validation_dir = './dataset/validation'



# data augmentation and data preprocessing for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# building models
from tensorflow.keras import layers, models

model = models.Sequential()

# Convolutional base
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# flatenning then output to feed it into dense layer
model.add(layers.Flatten())

# adding dense layer
model.add(layers.Dense(512, activation='relu'))

# prevent overfitting
model.add(layers.Dropout(0.5))

# output 
model.add(layers.Dense(1, activation='sigmoid'))

# compiling model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Total number of batches in the training set
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)   # Total number of batches in the validation set
)

val_loss, val_acc = model.evaluate(validation_generator)
model.save("cat_vs_dog.h5")
print(f"Validation Accuracy: {val_acc*100:.2f}%")