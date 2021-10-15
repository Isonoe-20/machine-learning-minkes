"""
Audio Dataset
By: Jordan Williams
"""

# System Imports
# Installed Imports
import librosa

#Custom Imports
from .GoogleDataset import GoogleDataset

class AudioDataset(GoogleDataset):
    """
    AudioDataset(client, bucketid, prefix)
    A Dataset object specifically for audio files
    stored in a Google Cloud Storage Bucket
    Methods:
        __getitem__(idx, tmpCount): Gets a numpy array of data stored in an
            audio file and returns it along with the sample rate.
    """
    def __init__(self, client, bucketid, prefix):
        super().__init__(client, bucketid, prefix)

    def __getitem__(self, idx, tmpCount=0):
        TEMPORARY_AUDIO_FILE = f"tmp{tmpCount}.wav"
        with open(TEMPORARY_AUDIO_FILE, 'wb') as file_obj:
            self.blobs[idx].download_to_file(file_obj)
        audio, sample_rate = librosa.load(TEMPORARY_AUDIO_FILE)
        # trackLength = len(audio) / sample_rate # Calculate the length of track using samples / sample rate
        return audio, sample_rate