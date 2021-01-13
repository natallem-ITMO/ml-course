import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    # (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    # Normalize
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return ( (train_images, train_labels), (test_images, test_labels))

def get_model():
    model =  models.Sequential()
    model.add(layers.Conv2D(filters = 20, kernel_size = 5, strides = 1, activation = 'tanh',
                            input_shape = (28,28,1), padding = 'same'))
    model.add(layers.AveragePooling2D(pool_size = 2, strides = 2, padding = 'valid'))
    model.add(layers.Conv2D(filters = 50, kernel_size = 5, strides = 1,activation = 'tanh',
                            padding = 'valid'))
    model.add(layers.AveragePooling2D(pool_size = 2, strides = 2, padding = 'valid'))

    model.add(layers.Flatten())
    model.add(layers.Dense(units = 500, activation = 'tanh'))
    model.add(layers.Dense(units = 10, activation = 'softmax'))
    model.summary()
    return model

def test_model(model, train_images, train_labels, test_images, test_labels):
    opt = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=20,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nТочность на проверочных данных:', test_acc)
    return model


def get_matrix_and_images(model, test_images, test_labels, class_num):
    predictions = model.predict(test_images)
    matrix = np.zeros((class_num, class_num), dtype = np.int32)
    probs = np.zeros((class_num, class_num))
    index = [[-1 for i in range(class_num)] for j in range(class_num)]

    for i in range(len(predictions)):
        predict = np.argmax(predictions[i])
        actual = test_labels[i]
        matrix[predict][actual] += 1
        prev_probs = probs[actual]
        cur_probs = predictions[i]
        for (j,(pr, cur)) in enumerate(zip(prev_probs, cur_probs)):
            if (pr < cur) :
                probs[actual][j] = cur
                index[actual][j] = i
    return (matrix, probs, index)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show_results(matrix, dataset, indexes, num_classes):
    plt.figure(figsize=(num_classes+10, num_classes+10))
    for arr in matrix:
        print(arr)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.subplot(num_classes, num_classes,i*10 + j +1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(dataset[indexes[i][j]], cmap=plt.cm.binary)
            plt.xlabel("a:" + class_names[i] + " cl:" + class_names[j])
    plt.show()

num_classes = 10
(train_images, train_labels), (test_images, test_labels) = load_dataset()
model= get_model()
model = test_model(model, train_images, train_labels, test_images, test_labels)
(matrix, probs, index) = get_matrix_and_images(model, test_images, test_labels, num_classes)
show_results(matrix, test_images, index, num_classes)



