"""

Audio Dataset Sanitiser
The object of this file is to take a dataset of big audio data.
For example the noaa pifsc dataset (8Tb) along with a list of interesting segments.
These segments are clipped and their features extracted for upload to a minio bucket system.

Date: 11/06/2021
By: Jordan Williams

"""

#Imports
import datetime
import csv

import librosa
import numpy as np

from GoogleDatasets import AudioDataset

REFERENCE_DATETIME = datetime.datetime(1970, 1, 1)

def get_audio_detections(
    file_path:str
    ) -> list[dict]:
    """
    get_audio_detections
    Will take a file path of a csv of detections and will create a useful
    list of dictionaries of data needed.
    returns detections
    """
    detections = []

    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        idx = 0
        for line in reader:
            # Change to DateTime Objects
            start_detection = datetime.datetime.strptime(
                line["DetectionTimeStart_UTC"],
                "%m/%d/%Y %H:%M:%S"
            )
            end_detection = datetime.datetime.strptime(
                line["DetectionTimeEnd_UTC"],
                "%m/%d/%Y %H:%M:%S"
            )
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

def get_extracted_detection(
    idx_audio_blob: int,
    start_datetimes: list[datetime.datetime],
    time_deltas: list[int],
    dataset: AudioDataset,
    detection: dict,
    first_file: bool = True
    ) -> tuple(np.array, np.array, int):
    """
    get_extracted_detection
    Downloads the file that has the existing detection within it.
    Finds the portion of the numpy array that corresponds with the detection.
    IF this goes beyond one file it will recursively go back and download the additional file.
    An assumption is made that any non detection in the audio file is noise.
    returns detection_clip, noise, sample_rate
    """
    # Download first audio file
    audio, sample_rate = dataset[idx_audio_blob]
    clip_length = len(audio) // sample_rate

    # Calculate distance through clip
    clip_start_time_seconds = int((
        start_datetimes[idx_audio_blob] - REFERENCE_DATETIME
    ).total_seconds())
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
        next_partial_clip, noise, sample_rate = get_extracted_detection(
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
        finish = int((
            abs(time_deltas[idx_audio_blob]) + int(detection["DetectionLength"])
        ) * sample_rate)
        return audio[start:finish], audio[finish:len(audio)], sample_rate

def get_feature_extractions(
    audio: np.array,
    sample_rate: int,
    prefix: str=""
    ) -> list:
    """
    get_feature_extractions
    Takes an audio array, this array consists of frequency points at a set sample rate
    It also takes the sample rate as this is needed for feature extraction.
    Prefix is used for seperating, unprocess, process, noise.
    A series of feature extraction is applied to the audio file.
    This affects the time-series nature of the data.
    returns a list of feature extractions from the audio.
    """
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
