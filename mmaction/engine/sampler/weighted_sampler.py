from torch.utils.data import BatchSampler, WeightedRandomSampler
from mmengine.registry import DATA_SAMPLERS
from typing import Optional

@DATA_SAMPLERS.register_module()
def weighted_sampler(dataset, seed:Optional[bool]=None, num_samples:Optional[int]=None):
    return WeightedRandomSampler(dataset.weights, num_samples=len(dataset) if not num_samples else num_samples)

@DATA_SAMPLERS.register_module()
def batch_sampler(sampler,batch_size,drop_last):
    return BatchSampler(sampler,batch_size,drop_last)