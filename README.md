# Machine Learning for Minkes
### By: Jordan Williams
Associated code for Part 3 Individual Project undertaken by students.
This code base is comprised of a few key components each of which will be documented here with example usage.

### GoogleDatasets
One of the first developments that was necessary to be made was an interface for the Google Cloud Storage system.
As it was known that the dataset was publically avaliable avoiding authentication would be ideal; due to the fact that it requires additional complicated setup the library was built around using an anonymous client.
- GoogleDatasets
    - GoogleDataset(client, bucket, prefix)
    - AudioDataset(client, bucket, prefix)

client - google.cloud.storage.Client object (can be either anonymous or authorised)
bucket - str object (bucket name)
prefix - str object (if you want a subfolder within the bucket if not default /)

AudoDataset has a *getitem* method, this will allow a specific blob to be downloaded to a temporary file.
Example:
    dataset = AudioDataset(...)
    audio, sample_rate = dataset[0]

#### Dependencies:
- google-cloud-storage
- librosa
