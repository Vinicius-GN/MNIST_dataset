import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import math

# Getting data from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
final_image = x_train 

# Flatten the images from 28x28 to a 1D array of 784 elements (Normalization)
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0 #Pixels are in the range of 0-255, so we normalize them to 0-1
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) # One hot encoding means that we have a vector of 10 elements and the value of each index corresponds if thats the number or not
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) #Thats the ground truth for the output

# Defining constants
input_size = 784  # 28x28 pixels
no_classes = 10
batch_size = 100  # Batches of training data
total_batches = 1  # Total number of epochs to train the model initially

# Optimization Step
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),  # Adjust the input shape based on your data
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 classes for MNIST
    ])
    return model

#Create the model
model = create_model()

# Define the loss function
loss_object = tf.keras.losses.CategoricalCrossentropy()

# Create a dataset from the training data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=10000).batch(batch_size)

# Create a summary writer for training and testing
train_summary_writer = tf.summary.create_file_writer('')
test_summary_writer = tf.summary.create_file_writer('')

for epoch in range(total_batches):
    for batch_no, (train_images, train_labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(train_images, training=True)
            loss = loss_object(train_labels, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Log training loss
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch * total_batches + batch_no)

        # Log test accuracy every 10 batches
        if batch_no % 10 == 0:
            test_logits = model(x_test, training=False)
            test_loss = loss_object(y_test, test_logits)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(test_logits, axis=1)), tf.float32))
            print("Accuracy:", math.ceil(float(accuracy)*100)/100)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=epoch * total_batches + batch_no)
                tf.summary.scalar('accuracy', accuracy, step=epoch * total_batches + batch_no)

train_summary_writer.close()
test_summary_writer.close()

while (True):
    index = int(input("Digite um valor entre 0 e 59999:"))
    #Tratamento de erro para valores fora do range
    if index < 0 or index > 59999:
        print("Valor fora do range")
        quit()
    img = final_image[index]
    img_flat = img.flatten() 
    plt.imshow(img_flat.reshape(28,28), cmap="Greys")
    img_flat = np.reshape(img_flat, (1, -1))

    logits_show = model(img_flat, training=False)
    plt.title(f"The output is {tf.argmax(logits_show, axis=1)}")
    plt.show()

