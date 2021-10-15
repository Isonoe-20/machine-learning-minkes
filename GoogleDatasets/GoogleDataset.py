class GoogleDataset:
    def __init__(self, client, bucketid, prefix=""):
        """
        GoogleDataset(client, bucketid, prefix)
        Creates a Google Dataset object for a Google Cloud Storage bucket.
        Creates some basic paramaters and methods for interacting with buckets
        Methods:
            __len__(): Returns amount of objects in given bucket
        """
        self.client = client
        self.bucket = self.client.get_bucket(bucketid)
        self.blobs = list(self.bucket.list_blobs(prefix=prefix))

    def __len__(self):
        return len(self.blobs)