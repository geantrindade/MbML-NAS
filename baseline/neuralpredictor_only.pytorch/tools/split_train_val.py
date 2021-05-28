import numpy as np
import h5py


def denoise_nasbench(metrics, threshold=0.8):
    val_metrics = metrics[:, -1, :, -1, 2]
    index = np.where(val_metrics[:, 0] > threshold)
    return index[0]


with h5py.File("data/nasbench.hdf5", mode="r") as f:
    total_count = len(f["hash"][()])
    metrics = f["metrics"][()]
result = dict()
for seed in [0, 1, 10, 42, 100, 123, 666, 1000, 1234, 12345]:
    random_state = np.random.RandomState(seed)
    for n_samples in [43, 86, 129, 172, 344, 860]:
        split = random_state.permutation(total_count)[:n_samples]
        label = str.format('%d_%d' % (seed, n_samples))
        result[label] = split

# >91
'''
valid91 = denoise_nasbench(metrics, threshold=0.91)
for n_samples in [172, 334, 860]:
    result["91-" + str(n_samples)] = np.intersect1d(result[str(n_samples)], valid91)
result["denoise-91"] = valid91
'''
#result["denoise-80"] = denoise_nasbench(metrics)
np.savez("data/train.npz", **result)
