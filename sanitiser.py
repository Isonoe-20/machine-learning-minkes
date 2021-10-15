"""

Audio Dataset Sanitiser
The object of this file is to take a dataset of big audio data.
For example the noaa pifsc dataset (8Tb) along with a list of interesting segments.
These segments are clipped and their features extracted for upload to a minio bucket system.

Date: 11/06/2021
By: Jordan Williams

"""
# Imports
import csv
import datetime
from typing import List, Dict

import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import storage
import noisereduce as nr
from tqdm import tqdm

from GoogleDatasets import AudioDataset
import minioClient

REFERENCE_DATETIME = datetime.datetime(1970, 1, 1)

def get_AudioDetections(
    filePath:str
    ) -> list[dict]:
    detections = []

    with open(filePath, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        idx = 0
        for line in reader:
            # Change to DateTime Objects
            start_detection = datetime.datetime.strptime(line["DetectionTimeStart_UTC"], "%m/%d/%Y %H:%M:%S")
            end_detection = datetime.datetime.strptime(line["DetectionTimeEnd_UTC"], "%m/%d/%Y %H:%M:%S")
            line["DetectionTimeStart_UTC"] = start_detection
            line["DetectionTimeEnd_UTC"] = end_detection

            # Calculate Detection length
            line["DetectionLength"] = (end_detection - start_detection).total_seconds()

            # Remove unecessary keys
            line.pop('DetectionTimeStart_ShipLocal')
            line.pop('Latitude')
            line.pop('Longitude')

            detections.append(line)
            idx += 1
    return detections

def get_ExtractedDetection(
    idx_audio_blob: int,
    start_datetimes: list[datetime.datetime],
    time_deltas: list[int],
    dataset: AudioDataset,
    detection: dict,
    first_file: bool = True
    ) -> np.array:
    # Download first audio file
        audio, sample_rate = dataset[idx_audio_blob]
        clip_length = len(audio) // sample_rate

        # Calculate distance through clip
        clip_start_time_seconds = int((start_datetimes[idx_audio_blob] - REFERENCE_DATETIME).total_seconds())
        clip_end_time_seconds = int(clip_start_time_seconds + clip_length)
        if first_file:
            detect_start_time_seconds = int(clip_start_time_seconds + abs(time_deltas[idx_audio_blob]))
        else:
            detect_start_time_seconds = int(clip_start_time_seconds)
        detect_end_time_seconds = int(detect_start_time_seconds + int(detection["DetectionLength"]))

        # Clip Detection from Array
        start = int(abs(time_deltas[idx_audio_blob]) * sample_rate)

        # Does Length of Clip excede EOF
        if clip_end_time_seconds <= detect_end_time_seconds:
            print("Clip in next File")
            idx_audio_blob += 1
            detection["DetectionLength"] = detect_end_time_seconds - clip_end_time_seconds
            next_partial_clip, noise, sample_rate = get_ExtractedDetection(
                idx_audio_blob,
                start_datetimes,
                time_deltas,
                dataset,
                detection,
                first_file=False
            )
            return np.concatenate([audio[start:len(audio)], next_partial_clip]), noise, sample_rate
        else:
            print("Clip in this File")
            finish = int((abs(time_deltas[idx_audio_blob]) + int(detection["DetectionLength"])) * sample_rate)
            return audio[start:finish], audio[finish:len(audio)], sample_rate

def get_FeatureExtractions(
    audio: np.array,
    sample_rate: int,
    prefix: str=""
    ) -> list:
    features = []

    # Save audio for later use
    np.save(f"./{prefix}/audio.npy", audio)

    # STFT
    # Base for contrast and chroma calculations
    stft = np.abs(librosa.stft(audio))

    #MFCC
    mfcc = np.mean(librosa.feature.mfcc(
        y = audio,
        sr = sample_rate,
        n_mfcc = 40
        ).T,
        axis=0)
    np.save(f"./{prefix}/mfcc.npy", mfcc)
    features.extend(mfcc)

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft,
        sr=sample_rate
        ).T,
        axis=0)
    np.save(f"./{prefix}/chroma.npy", chroma)
    features.extend(chroma)

    # Melspectogram
    spectogram = np.mean(librosa.feature.melspectrogram(
        y = audio,
        sr = sample_rate
    ).T,
    axis=0)
    np.save(f"./{prefix}/spectogram.npy", spectogram)
    features.extend(spectogram)

    # Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft,
        sr=sample_rate
        ).T,
        axis=0)
    np.save(f"./{prefix}/contrast.npy", contrast)
    features.extend(contrast)

    return features

# Main
if __name__ == "__main__":
    print("Main Thread Starting...")
    startIdx = 400
    clientminio =  minioClient.client

    client = storage.Client.create_anonymous_client()
    dataset = AudioDataset(client, 'noaa-pifsc-bioacoustic', '1705/')
    detections = get_AudioDetections('detections.csv')
    idx = 0

    for detection in tqdm(detections):
        if idx >= startIdx:
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

            detection_clip, noise, sample_rate = get_ExtractedDetection(
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
            get_FeatureExtractions(noise, sample_rate, prefix="output/noise")
            get_FeatureExtractions(detection_clip, sample_rate, prefix="output/unprocessed")
            get_FeatureExtractions(processed_clip, sample_rate, prefix="output/processed")

            # Upload images
            listOfFiles = glob.glob("./output/**/*.npy")
            splitListOfFiles = [file.split('/') for file in listOfFiles]
            for filePath in splitListOfFiles:
                minioPath = filePath[2:] # Remove unecessary file path elements
                minioPath.insert(1, f"sample{idx}") # Add new elements
                minioPath = '/'.join(minioPath) # Join Elements
                result = clientminio.fput_object(
                    'noaa-pifsc-bioacoustic',
                    minioPath,
                    "/".join(filePath)
                )
        idx += 1