"""

Audio Dataset Sanitiser
The object of this file is to take a dataset of big audio data.
For example the noaa pifsc dataset (8Tb) along with a list of interesting segments.
These segments are clipped and their features extracted for upload to a minio bucket system.

Date: 11/06/2021
By: Jordan Williams

"""
# Imports
import datetime
import glob

import librosa
import librosa.display
import numpy as np
from google.cloud import storage
import noisereduce as nr
from tqdm import tqdm

from GoogleDatasets import AudioDataset
from sanitiser_utils import (
    get_feature_extractions,
    get_extracted_detection,
    get_audio_detections
)
import minioClient

# Main
if __name__ == "__main__":
    print("Main Thread Starting...")
    STARTIDX = 400
    clientminio =  minioClient.client

    client = storage.Client.create_anonymous_client()
    dataset = AudioDataset(client, 'noaa-pifsc-bioacoustic', '1705/')
    detections = get_audio_detections('detections.csv')
    IDX = 0

    for detection in tqdm(detections):
        if IDX >= STARTIDX:
            start_datetimes = []
            time_deltas = []
            for blob in dataset.blobs:
                try:
                    start_datetime = datetime.datetime.strptime(blob.name[10:-8], "%Y%m%d_%H%M%S")
                except:
                    print("Unrecognised File Name")
                time_delta = (start_datetime - detection["DetectionTimeStart_UTC"]).total_seconds()
                start_datetimes.append(start_datetime)
                time_deltas.append(time_delta)

            # Find audio file
            idx_audio_blob = np.argmin(abs(np.array(time_deltas)))

            detection_clip, noise, sample_rate = get_extracted_detection(
                idx_audio_blob,
                start_datetimes,
                time_deltas,
                dataset,
                detection
            )

            # Reduce Noise
            processed_clip = nr.reduce_noise(
                y = detection_clip,
                y_noise = noise,
                sr = sample_rate
                )

            # Trim
            processed_clip, _ = librosa.effects.trim(
                y = processed_clip,
                top_db = 20,
                frame_length = 512,
                hop_length = 64
            )

            # Get Features
            get_feature_extractions(noise, sample_rate, prefix="output/noise")
            get_feature_extractions(detection_clip, sample_rate, prefix="output/unprocessed")
            get_feature_extractions(processed_clip, sample_rate, prefix="output/processed")

            # Upload images
            listOfFiles = glob.glob("./output/**/*.npy")
            splitListOfFiles = [file.split('/') for file in listOfFiles]
            for file_path in splitListOfFiles:
                MINIOPATH = file_path[2:] # Remove unecessary file path elements
                MINIOPATH.insert(1, f"sample{IDX}") # Add new elements
                MINIOPATH = '/'.join(MINIOPATH) # Join Elements
                result = clientminio.fput_object(
                    'noaa-pifsc-bioacoustic',
                    MINIOPATH,
                    "/".join(file_path)
                )
        IDX += 1
