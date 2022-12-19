import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 
from PIL import Image
import imagehash


#  Constants ########################################################
learning_rate = 0.001
input_shape=(150, 150, 3)

DATA_PATH = './data/Coffee Bean.csv'
model_save_path = './model.h5'
model_dir = './models/'

def calculate_phashes(image_stream):
    img = Image.open(image_stream)
    phash = str(imagehash.phash(img))
    return phash

def data_preparation(dataset):
    phashes = []
    for path in dataset.filepaths:
        im_path = './data/' + path
        phashes.append(calculate_phashes(im_path))
    dataset['phashes'] = phashes
    dataset = dataset.drop_duplicates(subset=['phashes'], keep='first').reset_index(drop=True)
    return dataset

def make_model(learning_rate=0.001, input_shape=(150, 150, 3)):
    # Model ########################################################################
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    # Optimizer and loss ###########################################################
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()

    # Compilation ##################################################################
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

    return model

def train_model():
    # Data preparation #################################################################

    df = pd.read_csv(DATA_PATH)
    assert df.shape == (1600, 4)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    df_full_train = df[df.data_set == 'train']
    del df_full_train['data_set']
    df_test = df[df.data_set == 'test']
    del df_test['data_set']

    ramdom_state = 42
    df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=ramdom_state)

    df_train = data_preparation(df_train)
    df_val = data_preparation(df_val)

    train_gen = ImageDataGenerator(rescale=1./255)

    train_ds = train_gen.flow_from_dataframe(
        dataframe=df_train,
        directory="./data/",
        x_col="filepaths",
        y_col="labels",
        batch_size=32,
        shuffle=True,
        class_mode="categorical",
        target_size=(150, 150))

    val_ds = train_gen.flow_from_dataframe(
        dataframe=df_val,
        directory="./data/",
        x_col="filepaths",
        y_col="labels",
        batch_size=32,
        shuffle=False,
        class_mode="categorical",
        target_size=(150, 150))

    test_ds = train_gen.flow_from_dataframe(
        dataframe=df_test,
        directory="./data/",
        x_col="filepaths",
        y_col="labels",
        batch_size=32,
        shuffle=False,
        class_mode="categorical",
        target_size=(150, 150))

    # Model training #############################################################


    model = make_model(learning_rate=learning_rate, input_shape=input_shape)

    print("Model summary")
    model.summary()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint = keras.callbacks.ModelCheckpoint('./models/model_v_{epoch:03d}_{val_accuracy:.3f}.h5',
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True, 
                                                )

    model.fit(train_ds, 
                epochs=10, 
                validation_data=val_ds,
                callbacks=[checkpoint]
            )

    best_model_path = os.listdir(model_dir)[-1]
    model.load_weights(model_dir + best_model_path)

    # Model evaluation #####################################################
    print('Model evaluation')
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'Test accuracy: {test_acc:.3f}')


    # Data augmentation ######################################################
    train_gen_augm = ImageDataGenerator(rescale=1./255,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')

    train_ds_augm = train_gen_augm.flow_from_dataframe(
        dataframe=df_train,
        directory="./data/",
        x_col="filepaths",
        y_col="labels",
        batch_size=32,
        shuffle=True,
        class_mode="categorical",
        target_size=(150, 150))

    # Training with data augmented ##################################################

    checkpoint = keras.callbacks.ModelCheckpoint(
        './models/model_x_v_{epoch:03d}_{val_accuracy:.3f}.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True 
        )

    model.fit(train_ds_augm, 
                epochs=10, 
                validation_data=val_ds,
                callbacks=[checkpoint]
                    )


    best_model_path = os.listdir(model_dir)[-1]
    model.load_weights(model_dir + best_model_path)

    # Model evaluation #####################################################
    print('Model evaluation after training on augmented data')
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'Test accuracy: {test_acc:.3f}')

    # Saving model #########################################################
    model.save_weights(model_save_path)
    print(f'model saved at {model_save_path}')



if __name__ == "__main__":
    train_model()





