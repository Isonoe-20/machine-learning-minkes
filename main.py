"""
Main Model Trainer file
Should train various models based on feature extraction for audio recognition
Date: 11/10/2021
By: Jordan Williams
"""

import tensorflow.keras as keras
import numpy as np
import re

from tensorflow.keras import callbacks
import minioClient

from tqdm import tqdm
from data_loaders import FeatureDataLoader

def AlexNet(
    n_features: int
    ) -> keras.Model:
    model = keras.Sequential()

    optimiser = keras.optimizers.Adam(
        0.001,
        0.9,
        0.999,
        amsgrad=False
    )
    model.add(keras.layers.Dense(256, activation='relu', input_dim = n_features))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss = "binary_crossentropy",
        optimizer = optimiser,
        metrics = ["accuracy"]
    )

    return model

if __name__ == "__main__":
    client = minioClient.client
    bucket = "noaa-pifsc-bioacoustic"
    path = ""
    label_names = ["noise", "processed"]
    files = []
    labels = []
    """
    Will take a Minio Client and a bucket in which to find the dataset
    A list of labels will also need to be supplied.
    This function will form a list of paths that
    correspond to the extracted features.
    In form like this:
    samples = [
            [bucket/label1/sample1/...1.npy,
            ...,
            bucket/label1/sample1/...n.npy,],
            [...],
            [bucket/label1/samplen/...1.npy,
            ...,
            bucket/label1/samplen/...n.npy,],
            ...,
            [bucket/labeln/sample1/...1.npy,
            ...,
            bucket/labeln/sample1/...n.npy,],
            ...]
        ]
    """
    for count, label in enumerate(label_names):
        samples = client.list_objects(
            bucket,
            prefix = f"{path}{label}/"
        )
        for sample in tqdm(samples):
            objects = client.list_objects(
                bucket,
                prefix = sample.object_name
            )
            feature_files = []
            for obj in objects:
                if not re.search(r'audio', obj.object_name):
                    feature_files.append(obj.object_name)
            files.append(feature_files)
            labels.append(count)

    joined_lists = list(zip(files, labels))
    np.random.shuffle(joined_lists)
    files , labels = zip(*joined_lists)

    ten_percent = int(np.floor(len(labels) / 10))
    train = FeatureDataLoader(
        client,
        bucket,
        files[0 : 8 * ten_percent],
        labels[0 : 8 * ten_percent],
        shuffle=True
    )
    val = FeatureDataLoader(
        client,
        bucket,
        files[(8 * ten_percent)+1 : 9 * ten_percent],
        labels[(8 * ten_percent)+1 : 9 * ten_percent],
        shuffle=True
    )
    test = FeatureDataLoader(
        client,
        bucket,
        files[(9 * ten_percent)+1 : int(len(labels))],
        labels[(9 * ten_percent)+1 : int(len(labels))],
        shuffle=True
    )

    model = AlexNet(187)
    history = None
    fpath = "./model.hdf5"
    checkpointer = keras.callbacks.ModelCheckpoint(
        fpath,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    samples, labels = train[0]
    print(samples.shape)
    print(labels.shape)
    history = model.fit(
        train,
        epochs = 30,
        validation_data=val,
        callbacks=[checkpointer],
        verbose = 2
    )