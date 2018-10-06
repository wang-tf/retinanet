import os
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger

from callbacks import RedirectModel
from callbacks.eval import Evaluate


def makedirs(path):
    """make a new dir"""
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def create_callbacks(model, training_model, prediction_model, validation_generator, configs):
    """add some callbacks"""
    model = training_model  # maybe model is the same as training_model, so here we use training_model

    callbacks = []

    # add loss callback
    if configs['Train']['reduce_lr']:
        loss_callback = ReduceLROnPlateau(monitor=configs['Train']['val_monitor'],
                                          factor=configs['Train']['lr_factor'],
                                          patience=3,
                                          verbose=1, mode='auto', epsilon=0.0001, cooldown=0,
                                          min_lr=configs['Train']['min_lr'])
        callbacks.append(loss_callback)

    # save the model
    if configs['Train']['save_snapshots']:
        # ensure directory created first; otherwise h5py will error after epoch.
        cwd = os.getcwd()
        snapshot_path = os.path.join(cwd, 'logs/'+configs['Name']+'/snapshots')
        if not os.path.exists(snapshot_path):
            makedirs(snapshot_path)
        h5_name = '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=configs['Train']['backbone'],
                                                                      dataset_type=configs['Dataset']['dataset_type'])
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(snapshot_path, h5_name), verbose=1, period=configs['Train']['period'])
        checkpoint_callback = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint_callback)

    if configs['Train']['tensorboard']:
        cwd = os.getcwd()
        log_dir = os.path.join(cwd, 'logs/'+configs['Name']+'/tensorboard')
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=configs['Train']['batch_size'])
        callbacks.append(tensorboard_callback)
    else:
        tensorboard_callback = None

    if configs['Train']['evaluation'] and validation_generator:

        if configs['Train']['save_path'] is not None and not os.path.exists(configs['Train']['save_path']):
            os.makedirs(configs['Train']['save_path'])
        evaluation = Evaluate(validation_generator, iou_threshold=0.5, score_threshold=0.05, max_detections=1000,
                                save_path=configs['Train']['save_path'], tensorboard=tensorboard_callback)

        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    return callbacks
