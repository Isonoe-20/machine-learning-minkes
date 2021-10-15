"""
Main Model Trainer file
Should train various models based on feature extraction for audio recognition
Date: 11/10/2021
By: Jordan Williams
"""

import tensorflow.keras as keras
import numpy as np
import re

from minioClient import client

class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        client,
        bucket,
        path,
        batch_size,
        label,
        shuffle
    ):
        self.client = client
        self.bucket = bucket
        self.path = path
        self.batch_size = batch_size
        self.label = label
        self.shuffle = shuffle

        # Calculate total files
        objects = self.client.list_objects(
            self.bucket,
            prefix=self.path
        )
        self.object_count = 0
        """self.objects = []
        for obj in objects:
            self.objects.append(obj.object_name)"""
        """
        loop Path/s

        minio path layout
        bucket/labelx/samplex/audio.npy
        bucket/labelx/samplex/...npy
        samples = []
        labels = []
        for label in labels:
            features = []
            get_objects()
            for obj in objects:
                if not contains audio:
                    features.append(obj.name)
            samples.append(features)
            labels.append(label.index)

        expected result:
            samples = [
                [bucket/label1/sample1/...1.npy,
                ...,
                bucket/label1/sample1/...n.npy,],
                [...],
                [bucket/label1/samplen/...1.npy,
                ...,
                bucket/label1/samplen/...n.npy,],
                [...],
                [bucket/labeln/sample1/...1.npy,
                ...,
                bucket/labeln/sample1/...n.npy,],
                ...
            ]
            labels = [1, ..., 1, ..., n, ...]
        """

    def __len__(self):
        return int(np.floor(self.object_count / self.batch_size))

    def __getitem__(
        self,
        idx
    ):
        batchObjectList = []
        batchStart = idx * self.batch_size
        batchEnd = (idx + 1) * self.batch_size
        while (batchStart < batchEnd):
            batchObjectList.append(f"{self.path}sample{batchStart}/")
            batchStart += 1
        for objPath in batchObjectList:
            subObjects = self.client.list_objects(
                self.bucket,
                prefix=objPath
            )
            for subObject in subObjects:
                if(not(re.search(r'audio', subObject.object_name))):
                    print(subObject.object_name)
        return idx

def AlexNet(
    n_features: int,
    n_classes: int
    ) -> keras.Model:
    model = keras.Sequential

    optimiser = keras.optimizers.Adam(
        0.001,
        0.9,
        0.999,
        amsgrad=False
    )

    model.add(
        keras.Dense(256, activation='relu'),
        input_dim = n_features
    )
    model.add(keras.Dense(128, activation='relu'))
    model.add(keras.Dense(64, activation='relu'))
    model.add(keras.Dense(n_classes, activation='softmax'))

    model.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = optimiser,
        metrics = ["accuracy"]
    )

if __name__ == "__main__":
    processed = DataGenerator(
        client,
        'noaa-pifsc-bioacoustic',
        'processed/',
        8,
        1,
        True
    )

    print(len(processed))
    processed[1]