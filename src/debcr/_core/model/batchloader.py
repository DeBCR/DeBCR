import os
import numpy as np

# generate batch-iterbale dataloader
class BatchLoader:
    def __init__(self, data, batch_size: int = 32, apply_noise: bool = False):
        
        self.data = data
        self.batch_size = batch_size
        self.apply_noise = apply_noise

        self._label_in = "low"
        self._label_gt = "gt"
        self._patches = self.data[self._label_in].shape[0]
        
        
    def _apply_noise(self, data):
        """Applies (additive Gaussian) noise."""
        
        sigma = np.random.uniform(0.0, 0.05) # set random
        noise = np.random.normal(0, sigma, data.shape)
        
        return data + noise

    def _rescale_all(self, data, MIN=0, MAX=1):
        """Rescales each 2D data slice to [MIN,MAX], default: [0,1]."""
        
        data_resc = np.zeros(shape=data.shape)
        
        for slice_idx in range(data.shape[0]):
            data_slice = data[slice_idx, ...]
            data_resc[slice_idx, ...] = np.interp(data_slice, (data_slice.min(), data_slice.max()), (MIN, MAX))
            
        return data_resc
        
    def __len__(self):
        """Returns the number of batches."""
        return int(np.floor(self._patches / self.batch_size))
        
    def __next__(self):
        """Fetches the next (random) batch."""
        
        batch_indices = np.random.choice(self._patches, self.batch_size, replace=False)
        
        batch_in = self.data[self._label_in][batch_indices]
        batch_gt = self.data[self._label_gt][batch_indices]

        batch_in = np.expand_dims(batch_in, axis=-1)
        batch_gt = np.expand_dims(batch_gt, axis=-1)
        
        if self.apply_noise:
            batch_in = self._apply_noise(batch_in)
        
        batch_in = self._rescale_all(batch_in)
        batch_gt = self._rescale_all(batch_gt)

        return batch_in, batch_gt
