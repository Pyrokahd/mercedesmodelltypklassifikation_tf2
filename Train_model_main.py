"""
Create a model and train it on the crawled images.
Then tests the validation set and optional the train set.

All results are saved in _tmp folders (In this project directory)
and need to be cleared manually before training a new one!
"""
import Create_models
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
# just for predictions test
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# TODO before training: check current save path and which model is created
EVALUATE_TEST = False
AUGMENTATION = False
RESHAPE_DATA = False  # if data should be B C H W instead of B H W C

BATCH_SIZE = 128  # 64  # Previous tests before v7 was used 32
IMG_SIZE = (224, 224)
TRAIN_EPOCHS = 1
LEARNING_RATE = 0.0001  # 0.001 for transformer, 0.0001 for the others
# | 0.001 based on keras tutorial for transformer | 0.0001 based on tensorflow finetuning tutorial (and tests confirm)

# Path to the dataset to use (will be split in train and validation)
DATA_PATH = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"
# Path to the test data set (splitted before by the script "ImageAugmentation"
TEST_DATA_PATH = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt_TEST_SET"

PROJECT_PATH = os.getcwd()
print(f"Project path:   {PROJECT_PATH}")
MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, "resultdata")
print(f"MODEL_SAVE_PATH:   {MODEL_SAVE_PATH}")
SAVE_FOLDER = "EXPERIMENT_Save_A_MODEL/modelsaves"  # TODO! Change this folder for new network tests ! (bsp. v9_[name]/ ..)


CLASS_NAMES = ['A205', 'A207', 'A208', 'A209', 'A217', 'A238', 'C117', 'C118', 'C167', 'C204', 'C205', 'C207', 'C208',
                'C209', 'C215', 'C216', 'C217', 'C218', 'C219', 'C238', 'C253', 'C257', 'C292', 'CL203', 'H247', 'N293',
                'S202', 'S203', 'S204', 'S205', 'S210', 'S211', 'S212', 'S213', 'T245', 'T246', 'V167', 'W163', 'W164',
                'W166', 'W202', 'W203', 'W204', 'W205', 'W210', 'W211', 'W212', 'W213', 'W247', 'W461-463', 'WV140',
                'WV220', 'WV221', 'WV222', 'WV223', 'X117', 'X118', 'X156', 'X164', 'X166', 'X167', 'X204', 'X218',
                'X247', 'X253']

def load_datasets(dataset_path, augmentation_data: bool = False, categorical: bool = False):
    print("loading data")
    if augmentation_data:
        data_dir = dataset_path = dataset_path+"_Augmented"
    else:
        data_dir = dataset_path

    # if labels need to be categorical like: https://keras.io/examples/vision/swin_transformers/
    # But doesnt seem to be the case in swintranformer2
    if categorical:
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical")

        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical")
    else:
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE)

        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE)

    return train_dataset, validation_dataset

def normalize_prepare_data(_train_dataset, _validation_dataset):
    # Normalize train and validation
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_dataset = _train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = _validation_dataset.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset

def create_class_weights_dict(_train_dataset):
    """
    Creating a dict with label and class weight for class weights for training.
    DOESNT WORK WITH ONE HOT ENCODED (Categorical) LABELS IN TRAIN SET!
    :param _train_dataset:
    :type _train_dataset:
    :return:
    :rtype:
    """
    # validation_dataset # PrefetchDataset
    allTargets = np.array([])

    for images, targets in _train_dataset:
        targets = targets.numpy()
        allTargets = np.concatenate((allTargets, targets))

    anzahl_alle_train_daten = len(allTargets)

    unique, counts = np.unique(allTargets, return_counts=True)
    # print(unique)
    # print(counts)

    class_weights = {}

    average_class_size = anzahl_alle_train_daten / 65

    for i in range(len(counts)):
        class_weights[i] = (1 / counts[i]) * average_class_size

    return class_weights

def plot_training_history(_path, _history):
    acc = _history.history['accuracy']
    val_acc = _history.history['val_accuracy']

    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    # plt.ylim([0,4.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(_path, "AccruracyAndLoss.png"))
    #plt.show()

def predict_on_data(_dataset, _model):
    """
    Returns an np array with the true labels (as indecies) and another for the predicted labels.
    :param _dataset:
    :type _dataset:
    :param _model:
    :type _model:
    :return:
    :rtype:
    """
    target_array = np.array([])
    prediction_array = np.array([])

    # iterates batch wise over validation dataset
    for images, targets in _dataset:
        # y = model(image, training=False)  # Or y = model.predict(image)
        y = _model.predict(images)

        # turn logits into normalized prediction
        y = tf.nn.softmax(y)  # shape is (32, 65) 32 rows with 65 columns each column is one class

        # Get predicted label (max value index of each of the 32 rows)
        y = np.argmax(y, axis=1)

        targets = targets.numpy()
        # print(target.shape)  # 32 , liste mit 32 einträgen

        target_array = np.concatenate((target_array, targets))
        prediction_array = np.concatenate((prediction_array, y))

    target_array = target_array.astype(int)
    prediction_array = prediction_array.astype(int)

    return target_array, prediction_array

def plot_class_distribution(_path, _dataset):
    allTargets = np.array([])

    for images, targets in _dataset:
        targets = targets.numpy()
        allTargets = np.concatenate((allTargets, targets))

    unique, counts = np.unique(allTargets, return_counts=True)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 2, 1])
    ax.bar(CLASS_NAMES, counts)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(_path, 'classDistribution.png'), bbox_inches="tight")
    #plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, _path=""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(30,30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(_path, 'confusion_matrix_blue.png'))

def plot_all_results_for_data(_path, _model, _dataset):
    # To evaluate
    print("Evaluate")
    result = _model.evaluate(_dataset)
    eval_results = dict(zip(_model.metrics_names, result))
    print(eval_results)
    print(20*"-")

    #class dist
    plot_class_distribution(_path, _dataset)
    plt.close(plt.gcf())

    # prection
    target_array, prediction_array = predict_on_data(_dataset, _model)

    ##Confusion Matrix
    cm = confusion_matrix(y_true=target_array, y_pred=prediction_array)

    print("first confusion matrix")
    # 1
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(38, 38))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(_path, 'confusion_matrix.jpg'))
    #plt.show()
    plt.close(plt.gcf())

    print("second confusion matrix")
    #2
    plt.figure(figsize=(14, 10))
    plt.matshow(cm, fignum=1)  # weil matshow sonst seine eigene figure erstellt muss hier auf die oben erstellte referenziert werden
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')

    # alternativ kann man die aktuelle figure abgreifen und ich nachhinein skalieren
    # fig = plt.gcf()  # get current figure
    # fig.set_size_inches(18.5, 10.5)

    plt.savefig(os.path.join(_path, 'confusion_matrix_matshow.png'))
    #plt.show()  # creates new fig, deshalb vorher speichern
    plt.close(plt.gcf())

    print("third confusion matrix")
    #3
    plot_confusion_matrix(cm, CLASS_NAMES, _path=_path)

def make_single_test_prediction(_model):
    """
    To test if the model can predict or some error occurs.
    Path to testimage is hardcoded.
    :param _model:
    :type _model:
    :return:
    :rtype:
    """
    imagename = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt/W213/" + "id-2011-image0-W213.jpg"
    print("PREDICTION TEST")
    image = load_img(imagename, target_size=(224, 224))
    input_arr = img_to_array(image)
    input_arr *= 1 / (255.0)  # normalize same way as in training
    # input_arr = input_arr/(255.0)
    # print(input_arr)

    image_tensor = np.array([input_arr])
    image_tensor = tf.convert_to_tensor(image_tensor)
    print(f"my shape before: {image_tensor.shape}")

    #If shape B H H W is needed:
    # image_tensor = tf.reshape(image_tensor, [1, 3, 224, 224])
    #print(f"my shape After: {image_tensor.shape}")

    output = _model.predict(image_tensor)
    ##print(output)
    y = tf.nn.softmax(output)  # shape is (32, 65) 32 rows with 65 columns each column is one class
    # Get predicted label (max value index of each of the 32 rows)
    y = np.argmax(y, axis=1)
    #print(y)

    print(f"Predicted Class: {CLASS_NAMES[33]}")
    print("PREDICTION TEST END")

# so it is not called by importing
if __name__ == "__main__":
    categorical_data = False  # Default False for all models except maybe convnext
    res_net_model = True

    ## Create Model

    # TODO change save folder! to a new one before every training
    if res_net_model:
        print("creating model:")
        #Resnet
        model = Create_models.create_ResNet(resnetType='ResNet50V2', dropoutrate=0.4, base_layer_to_finetune=40)
        #VGG

    else:
        print("creating model:")
        #Transformer

        ###############
        ## Variant 1 ##
        # ! ERROR: while training !
        #model = create_models.create_swin_transformer_model()
        #RESHAPE_DATA = True
        # error: Cant convert None to Tensor

        ###############
        ## Variant 2 ##
        # ! ERROR: while training !
        #model = create_models.create_convnext_model()
        #categorical_data = True
        # error ValueError: Dimensions must be equal, but are 65 and 7 ...

        ###############
        ## Variant 3 ##
        # ! bad training with set hyperparameter !
        model = Create_models.create_swin_transformer_model_version3()  # which size is defined in the called func

        # SwinTransformerGit_1 (variant 1) needs B C H W
        if RESHAPE_DATA:
            x = tf.zeros((1, 3, 224, 224), dtype=tf.float64)
            y = model(x)
            print(y.shape)
        # All other Models and Variants need B H W C
        else:
            # put 1 input through net so the weights are created
            x = tf.zeros((1, 224, 224, 3), dtype=tf.float64)
            y = model(x)
            print(y.shape)


    print("Model Summary: ")
    model.summary()

    print(f"trainable variables: {len(model.trainable_variables)}")


    ## Load Dataset
    train_dataset, validation_dataset = load_datasets(DATA_PATH, AUGMENTATION, categorical_data)

    ## Normalize data
    train_dataset, validation_dataset = normalize_prepare_data(train_dataset, validation_dataset)


    ## Set training Parameters
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    initial_epochs = TRAIN_EPOCHS

    callbacks = [
        # This callback saves a SavedModel every 2000 batches.
        # We include the training loss in the saved model name.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(PROJECT_PATH, "checkpoints") + "/ckpt-loss={loss:.2f}",
            save_freq=2000,
            # monitor='val_accuracy',
            mode='max'
            # save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    ]

    # load again just for class_weights since it doesnt work with one hot encoded labels
    if categorical_data:
        train_dataset_cw, validation_dataset_cw = load_datasets(DATA_PATH, AUGMENTATION, categorical=False)
    else:
        train_dataset_cw = train_dataset
    # CREATE CLASS WEIGHTS
    class_weights = create_class_weights_dict(train_dataset_cw)

    # shape sanity check
    #image_batch, label_batch = next(iter(train_dataset))
    #print(f"Before Size of Image batch: {image_batch.shape}")

    # If we need Data in B C H W this reshapes all data sets
    if RESHAPE_DATA:
        ## size check of data (swin model 1 needs B C H W)  with H and W = 224
        ## but i have data of (B H W C)
        ## Reshape all images
        reshape_layer = tf.keras.layers.Reshape((3, 224, 224))  # batch is done in the dataset not in the reshape method
        # reshape via layer
        train_dataset = train_dataset.map(lambda x, y: (reshape_layer(x), y))
        validation_dataset = validation_dataset.map(lambda x, y: (reshape_layer(x), y))

        image_batch, label_batch = next(iter(train_dataset))
        print(f"Size of Image batch after lambda map: {image_batch.shape}")
        print("--shape check 2 end--")


    # Single Image prediction
    #make_test_prediction(model)


    ## Train
    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset,
                        class_weight=class_weights,
                        callbacks=callbacks)

    ## train History
    print("stats saved to: ")
    val_path = os.path.join(PROJECT_PATH, "resultdata/_tmp_validationset_stats")  # for all further validation evals
    print(val_path)

    plot_training_history(val_path, history)
    plt.close(plt.gcf())  # close current plt (had some weird one with background plot and foreground Confusion matrix

    # Validation Evaluation
    print("start plotting...")
    plot_all_results_for_data(val_path, model, validation_dataset)
    print("done plotting, now saving...")

    ## SAVE MODEL IN ALL VARIANTS
        # load with:
        #folder = "C:/Users/Christian/PycharmProjects/InnovationsProjektWebScraper/resultdata/Resnet50v2_4_weighted/fineTuned"
        #model = tf.keras.models.load_model(folder + "/myModel_fineTuned_kerasSave")
        #model.trainable = False

    #1
    print(f'SAVING MODEL TO:  {MODEL_SAVE_PATH + "/" + SAVE_FOLDER + "/myModel_fineTuned"}')
    ## model save method save as folder  (as .h5 didnt work)
    model.save(MODEL_SAVE_PATH + "/" + SAVE_FOLDER + "/myModel_fineTuned")
    # model = tf.keras.models.load_model(modelsavepath+"/"+folder+"/myModel_fineTuned")

    print("save 2")
    #2
    tf.keras.models.save_model(model, MODEL_SAVE_PATH + "/" + SAVE_FOLDER + "/myModel_fineTuned_kerasSave")
    # model = tf.keras.models.load_model(modelsavepath+"/"+folder+"/myModel_fineTuned_kerasSave")

    print("save 3")
    #3
    ## saved_model save method
    tf.saved_model.save(model, MODEL_SAVE_PATH + "/" + SAVE_FOLDER + "/myModel_fineTuned_TfSave")
    # model = tf.saved_model.load(modelsavepath+"/"+folder+"/myModel_fineTuned_saveV2")

    print("save 4")
    #4
    ## only save weights
    model.save_weights(MODEL_SAVE_PATH + "/" + SAVE_FOLDER + '/weights_myModel_fineTuned')
    ## new_model = CREATE MODEL
    # new_model.load_weights(modelsavepath+"/"+folder+'myMode_fineTuned_weights.h5')


    ## Optional Train Test
    # Load dataset
    if EVALUATE_TEST:
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            TEST_DATA_PATH,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE)

        # Preprocessing
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

        AUTOTUNE = tf.data.AUTOTUNE
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
        print(test_dataset)

        test_path = os.path.join(PROJECT_PATH, "resultdata/_tmp_testset_stats")
        plot_all_results_for_data(test_path, model, test_dataset)

