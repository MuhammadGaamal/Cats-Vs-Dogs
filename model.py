from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3),use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

# Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, (3, 3),input_shape = (50, 50, 3), use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 128,kernel_initializer='normal' ,use_bias=False))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))
model.add(Dense(units = 1, kernel_initializer='normal' ,use_bias=False))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))
# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_data',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

model.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 5,
                         validation_steps = 2000)

model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./model.h5")
print("saved model..! ready to go.")

