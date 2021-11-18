"""
Data Generator
for bringing in the noaa pisfc sanitised dataset from minio.
How we want the data returned:
    samples = [45.6, 34.2, ..., n]
    labels = [label, label, ..., n]
    [samples, labels]
"""
# Imports

import numpy as np
from tensorflow import keras

class FeatureDataLoader(keras.utils.Sequence):
    """
    A data loader for the nooa pifsc sanitised dataset
    """
    def __init__(
        self,
        client,
        bucket,
        files,
        labels,
        batch_size = 8,
        shuffle = False
        ) -> None:
        """
        Initiates class with a list of files and labels.
        It also defines if the dataset is shuffled at the end of an epoch
        as well as batch sizes.
        """
        self.files = files
        self.labels = labels
        self.client = client
        self.bucket = bucket
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """
        Returns the number of batches that this DataGenerator
        is capable of.
        """
        length = int(np.floor(len(self.files) / self.batch_size))
        return length

    def __getitem__(
        self,
        idx: int
    ):
        """
        Gets the current batch requested by the index
        This is determined by
        [idx * batch_size, (idx+1) * batch_size]
        If the objects are setup as a list of files
        These should be downloaded at runtime not beforehand.
        A Cache Sytem is a future nice to have.
        Returns:
            samples = [45.6, 34.2, ..., n]
            labels = [label, label, ..., n]
            [samples, labels]
        """
        batch = self.files[
            idx * self.batch_size:
            (idx + 1) * self.batch_size
        ]

        batch_samples = []
        batch_labels = []
        for count, sample in enumerate(batch):
            features = []
            for file in sample:
                self.client.fget_object(
                    self.bucket,
                    file,
                    "tmp.npy"
                )
                with open("tmp.npy", "rb") as file_obj:
                    data_request = np.load(file_obj)
                features.extend(data_request)
            batch_samples.append(features)
            batch_labels.append(self.labels[
                (idx * self.batch_size) + count
                ])
        return np.array(batch_samples), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle == True:
            joined_lists = list(zip(self.files, self.labels))
            np.random.shuffle(joined_lists)
            self.files , self.labels = zip(*joined_lists)
