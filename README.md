# Machine Learning for Minkes
### By: Jordan Williams
Associated code for Part 3 Individual Project undertaken by students.
This code base is comprised of a few key components each of which will be documented here with example usage.
Designed for Python 3.9

### GoogleDatasets
One of the first developments that was necessary to be made was an interface for the Google Cloud Storage system.

As it was known that the dataset was publically avaliable avoiding authentication would be ideal; due to the fact that it requires additional complicated setup the library was built around using an anonymous client.
- GoogleDatasets
    - GoogleDataset(client, bucket, prefix)
    - AudioDataset(client, bucket, prefix)
```
Dataset(
    client, - google.cloud.storage.Client object (can be either anonymous or authorised)
    bucket, - str object (bucket name)
    prefix - str object (if you want a subfolder within the bucket if not default /)
)
```
AudoDataset has a *getitem* method, this will allow a specific blob to be downloaded to a temporary file.

Example:
```
    dataset = AudioDataset(...)
    audio, sample_rate = dataset[0]
```

#### Dependencies:
- google-cloud-storage
- librosa

### NOAA-PIFSC Sanitiser
The public avaliable NOAA-PIFSC dataset is of a large size (8Tb+) and as such would be rather large to load into memory or even store locally. This leads to speed issues. There are roughly 11,800 detections of Minke Whales in the dataset; an unknown number of 1 minute clips spanning atleast 3 days are included. A solution that is proposed is to extract these samples from the clips this will reduce the size of the dataset down to the 11,800 clips. For later purposes it is worth also saving noise samples at the same time so we have an equal sample size for both classes. It's also allowed for the unprocessed audio to be saved too. The file format of the outputted audio file is as a numpy array or *.npy*. These extracted clips are roughly 1-3 MiB this is a ~99% decrease in size. However we are now saving 3 arrays per sample. Totalling a minimum of 106,200Mb or 106Gb this is still too large for most resonable systems. To reduce this further it was decided to perform the feature extraction on the clips at the same time; this now reduces the size of each feature extraction to 100-600 Bytes. Giving the final minimum total size of all the data as 42,480,000 Bytes or 42.48 Mb a much more resonable size to load into memory.