# Swin Transformer
import argparse
import tensorflow as tf

# from https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow :
from SwinTransformerGit_1.config import get_config
from SwinTransformerGit_1.models.build import build_model

# from https://github.com/bamps53/convnext-tf :
from convnextTFmasterGit.models.convnext_tf import create_model

# from https://github.com/rishigami/Swin-Transformer-TF :
from SwinTransformerGit_2.model import SwinTransformer

#expects image of size 224x224
CUSTOM_NUM_CLASSES = 65

##############
# ResNet model from tensorflow applications
def create_ResNet(resnetType = 'ResNet50V2', dropoutrate = 0.2, base_layer_to_finetune = 40):
    """
    Creates a model with one of 3 resnet Types:
    'ResNet50V2' , 'ResNet101V2' , 'ResNet152V2'
    Adds globalAveragePooling -> Dropout -> Dense Prediction Layer
    to the base_model from ResNet and returns the stacked model.

    :param resnetType:  Which Resnet Version
    :type resnetType: String
    :param dropoutrate: dropout rate for the dropout layer
    :type dropoutrate: float
    :param base_layer_to_finetune: How many layers from the resnet are trainable (i.e. 40 = last 40 layers)
    :type base_layer_to_finetune: int
    :return: tensorflow keras model
    :rtype:
    """
    IMG_SIZE = (224, 224)
    IMG_SHAPE = IMG_SIZE + (3,)

    if resnetType=='ResNet50V2':
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
    elif resnetType=='ResNet101V2':
        base_model = tf.keras.applications.resnet_v2.ResNet152V2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
    else:
        base_model = tf.keras.applications.resnet_v2.ResNet152V2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(65)
    print("Number of layers in the base model: ", len(base_model.layers))

    # Define trainable layer
    fine_tune_at = len(base_model.layers) - base_layer_to_finetune  # 190-40 = 150 meaning first 150 are not trainable
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    # x = normalization_layer(inputs)
    x = base_model(inputs) # the "unwanted" layers are set to not trainable in the fine tuning settings above
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropoutrate)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model
##############


##############
# Swin Transformer 1 from: https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer weight conversion from PyTorch to TensorFlow', add_help=False)

    parser.add_argument(
        '--cfg',
        type=str,
        metavar="FILE",
        help='path to config file',
        default="configs/swin_base_patch4_window7_224.yaml"
    )
    parser.add_argument(
        '--include_top',
        type=int,
        help='Whether or not to include model head',
        choices={0, 1},
        default=0,
    )
    parser.add_argument(
        '--resume',
        type=int,
        help='Whether or not to resume training from pretrained weights',
        choices={0, 1},
        default=1,
    )
    parser.add_argument(
        '--weights_type',
        type=str,
        help='Type of pretrained weight file to load including number of classes',
        choices={"imagenet_1k", "imagenet_22k", "imagenet_22kto1k"},
        default="imagenet_1k",
    )

    args = parser.parse_args()
    config = get_config(args, include_top=bool(args.include_top))

    return args, config
def main_swintransformer(args, config):
    swin_transformer = build_model(config, load_pretrained=bool(args.resume), weights_type=args.weights_type)

    print(
        swin_transformer(
            tf.zeros([
                1,
                config.MODEL.SWIN.IN_CHANS,
                config.DATA.IMG_SIZE,
                config.DATA.IMG_SIZE
            ])
        )
    )

    return swin_transformer
def create_swin_transformer_model(base_trainable: bool = False):

    args, config = parse_option()
    swin_transformer_model_base = main_swintransformer(args, config)

    if not base_trainable:
        swin_transformer_model_base.trainable = False
    print("Number of layers in the base model: ", len(swin_transformer_model_base.layers))

    swin_transformer_model = tf.keras.Sequential([
        swin_transformer_model_base,
        tf.keras.layers.Dense(CUSTOM_NUM_CLASSES)
    ])

    return swin_transformer_model
##############


##############
# Convnext Net from: https://github.com/bamps53/convnext-tf
def create_convnext_model():
    #x = tf.zeros((1, 224, 224, 3), dtype=tf.float32)  # done in train_model_main
    basemodel = create_model('convnext_base_224', input_shape=(224, 224), num_classes=CUSTOM_NUM_CLASSES,
                         pretrained=True, include_top=False)

    basemodel.trainable = False
    print("Number of layers in the base model: ", len(basemodel.layers))

    swin_transformer_model_alt = tf.keras.Sequential([
        tf.keras.layers.Lambda(
            lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"),
            input_shape=[224, 224, 3]),
        basemodel,
        tf.keras.layers.Dense(CUSTOM_NUM_CLASSES)
    ])

    return swin_transformer_model_alt
    #out = model(x)  # (1, 1000)
##############


##############
# Swin Transformer 2 from: https://github.com/rishigami/Swin-Transformer-TF
def create_swin_transformer_model_version3():
    """
    This one uses this github https://github.com/rishigami/Swin-Transformer-TF
    :return:
    :rtype:
    """
    basemodel = SwinTransformer('swin_small_224', include_top=False, pretrained=True, use_tpu=False)
    basemodel.trainable = False
    print("Number of layers in the base model: ", len(basemodel.layers))

    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(
            lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"),
            input_shape=[224, 224, 3]),
        basemodel,
        tf.keras.layers.Dense(CUSTOM_NUM_CLASSES)  # activation='softmax'  no softmax we use from logits in loss?
    ])

    # example multiple layer by creating a model class that inherits form keras.Model
    # https://github.com/rishigami/Swin-Transformer-TF/issues/9  ,"call" is called by keras and tensorflow to calculate

    return model
##############