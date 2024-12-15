import numpy as np

class BatchSampler:
    """
    A utility for repeatedly sampling batches of indices from a set of samples.

    Behavior:
    - Indices from [0, num_samples-1] are shuffled each epoch if specified.
    - Subsequent calls to get_next() return consecutive mini-batches.
    - Once the end of an epoch is reached, it starts a new epoch (optionally reshuffling).
    Args:
        num_samples (int): Total number of samples.
        batchSize (int, optional): Default batch size. If not provided, must be specified when calling get_next().
        shuffle (bool): If True, indices are reshuffled at the start of each epoch.
    """

    def __init__(self, num_samples, batchSize=None, shuffle=True):
        self.num_samples = num_samples
        self.initial_batch_size = batchSize
        self.shuffle = shuffle

        # Initialize the index array and epoch counters
        self._indices = np.arange(num_samples)
        self._completed_epochs = 0
        self._pos_in_epoch = 0

        # Shuffle indices at the very beginning if requested
        if self.shuffle:
            np.random.shuffle(self._indices)

    @property
    def epochs_completed(self):
        """Returns the number of fully completed epochs."""
        return self._completed_epochs

    def get_next(self, batch_size=None):
        """
        Retrieve the next batch of indices.

        Args:
            batch_size (int, optional): Number of elements in the batch. Uses initial_batch_size if not set.

        Returns:
            np.ndarray: Array of indices for the requested batch.
        """
        if batch_size is None:
            batch_size = self.initial_batch_size

        if batch_size > self.num_samples:
            raise ValueError(f"batch_size={batch_size} is greater than num_samples={self.num_samples}.")

        start_idx = self._pos_in_epoch
        end_idx = start_idx + batch_size

        if end_idx <= self.num_samples:
            # Entire batch fits in the current epoch
            self._pos_in_epoch = end_idx
            return self._indices[start_idx:end_idx]
        else:
            # Crosses epoch boundary
            self._completed_epochs += 1

            # Take what remains in the current epoch
            samples_left = self.num_samples - start_idx
            partial_batch = self._indices[start_idx:self.num_samples].copy()

            # Shuffle for the new epoch if needed
            if self.shuffle:
                np.random.shuffle(self._indices)

            # Remaining part of the batch from the start of the new epoch
            self._pos_in_epoch = batch_size - samples_left
            new_chunk = self._indices[0:self._pos_in_epoch]

            # Combine the two parts
            return np.concatenate([partial_batch, new_chunk])