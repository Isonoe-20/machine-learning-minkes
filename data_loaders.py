"""
Data Generator
for bringing in the noaa pisfc sanitised dataset from minio.
How we want the data returned:
    samples = [45.6, 34.2, ..., n]
    labels = [label, label, ..., n]
    [samples, labels]
"""
# Imports
import re

import numpy as np
from tqdm import tqdm
from tensorflow import keras

import minioClient

class FeatureDataLoader(keras.utils.Sequence):
    """
    A data loader for the nooa pifsc sanitised dataset
    """
    def __init__(
        self,
        client: minioClient.client,
        bucket: str,
        path: str,
        label_names: list[str],
        batch_size = 8,
        shuffle = False
        ) -> None:
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
        self.client = client
        self.bucket = bucket
        self.label_names = label_names
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.files = []
        self.labels = []
        for count, label in enumerate(label_names):
            samples = self.client.list_objects(
                self.bucket,
                prefix = f"{path}{label}/"
            )
            for sample in tqdm(samples):
                objects = self.client.list_objects(
                    self.bucket,
                    prefix = sample.object_name
                )
                feature_files = []
                for obj in objects:
                    if not re.search(r'audio', obj.object_name):
                        feature_files.append(obj.object_name)
                self.files.append(feature_files)
                self.labels.append(count)

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
        return (batch_samples, batch_labels)

if __name__ == "__main__":
    generator = FeatureDataLoader(
        minioClient.client,
        "noaa-pifsc-bioacoustic",
        "",
        ["noise", "processed"]
    )
