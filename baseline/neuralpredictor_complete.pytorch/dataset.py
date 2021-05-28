import h5py
import numpy as np
from torch.utils.data import Dataset



'''
0 -> 0.23
1 -> 0.54
2 -> 0.86
3 -> 0.91
'''

thresholds = [0.23, 0.54, 0.86, 0.91]

class Nb101Dataset(Dataset):
    MEAN = 0.908192
    STD = 0.023961

    def __init__(self, split=None, exclude_split=[], epoch_set=0, seed=0, debug=False):
        self.hash2id = dict()
        
        self.EPOCH_SET = epoch_set
        self.ACC_THRESHOLD = thresholds[epoch_set]
        np.random.seed(int(seed))
        print('Epoch Set: %d \nACC_Threshold: %.2f \n' % (self.EPOCH_SET, self.ACC_THRESHOLD))
        
        with h5py.File("data/nasbench.hdf5", mode="r") as f:
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f["num_vertices"][()]
            self.trainable_parameters = f["trainable_parameters"][()]
            self.adjacency = f["adjacency"][()]
            self.operations = f["operations"][()]
            self.metrics = f["metrics"][()]
        self.random_state = np.random.RandomState(0)
        
        '''
        if len(exclude_split) != 0:
            exclude_split = np.load("data/train.npz")[str(exclude_split)]
        '''
        if split is not None and split != "all":
            sample_set = np.load("data/train.npz")[str(split)]
            select_set = np.isin(sample_set, exclude_split)
            self.sample_range = sample_set[~select_set] 

        else:
            sample_set = np.array(range(len(self.hash2id)))
            select_set = np.isin(sample_set, exclude_split)
            self.sample_range = sample_set[~select_set]

        self.debug = debug
        self.seed = seed

    def __len__(self):
        return len(self.sample_range)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def mean_acc(self):
        return np.mean(self.metrics[:, self.EPOCH_SET, self.seed, -1, 2])

    def std_acc(self):
        return np.std(self.metrics[:, self.EPOCH_SET, self.seed, -1, 2])

    @classmethod
    def normalize(cls, num):
        return (num - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, num):
        return num * cls.STD + cls.MEAN

    def resample_acc(self, index, split="val"):
        # when val_acc or test_acc are out of range
        assert split in ["val", "test"]
        split = 2 if split == "val" else 3
        for seed in range(3):
            acc = self.metrics[index, self.EPOCH_SET, seed, -1, split]
            #if not self._is_acc_blow(acc):
            return acc
        if self.debug:
            print(index, self.metrics[index, self.EPOCH_SET, :, -1])
            raise ValueError
        return np.array(self.MEAN)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    def _is_class_high_threshold(self, acc):
        return acc >= ACC_THRESHOLD #0.91


    def __getitem__(self, index):
        index = self.sample_range[index]

        self.seed = np.random.randint(0, 3)
        val_acc, test_acc = self.metrics[index, self.EPOCH_SET, self.seed, -1, 2:]
        val_avg = 0 
        test_avg = 0
        
        for seed_test_avg in [0, 1, 2]:
            val_acc, test_acc = self.metrics[index, self.EPOCH_SET, seed_test_avg, -1, 2:]
            val_avg += val_acc
            test_avg += test_acc
        
        val_acc = val_avg / 3
        test_acc = test_avg / 3

        n = self.num_vertices[index]
        ops_onehot = np.array([[i == k + 2 for i in range(5)] for k in self.operations[index]], dtype=np.float32)
        if n < 7:
            ops_onehot[n:] = 0.
        result = {
            "index": index,
            "num_vertices": n,
            "adjacency": self.adjacency[index],
            "operations": ops_onehot,
            "mask": np.array([i < n for i in range(7)], dtype=np.float32),
            "val_acc": val_acc,
            "test_acc": test_acc
        }
        if self.debug:
            self._check(result)
        return result


class DatasetPred(Dataset):
    MEAN = 0.908192
    STD = 0.023961

    def __init__(self, split=None, exclude_split=[], epoch_set=0, seed=0, debug=False):
        self.hash2id = dict()
        
        self.EPOCH_SET = epoch_set
        self.ACC_THRESHOLD = thresholds[epoch_set]
        np.random.seed(int(seed))
        print('Epoch Set: %d \nACC_Threshold: %.2f \n' % (self.EPOCH_SET, self.ACC_THRESHOLD))
        
        with h5py.File("data/nasbench.hdf5", mode="r") as f:
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f["num_vertices"][()]
            self.trainable_parameters = f["trainable_parameters"][()]
            self.adjacency = f["adjacency"][()]
            self.operations = f["operations"][()]
            self.metrics = f["metrics"][()]
        self.random_state = np.random.RandomState(0)
        
        sample_set = split #np.array(range(len(self.hash2id)))
        select_set = np.isin(sample_set, exclude_split)
        self.sample_range = sample_set[~select_set]

        self.debug = debug
        self.seed = seed

    def __len__(self):
        return len(self.sample_range)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def mean_acc(self):
        return np.mean(self.metrics[:, self.EPOCH_SET, self.seed, -1, 2])

    def std_acc(self):
        return np.std(self.metrics[:, self.EPOCH_SET, self.seed, -1, 2])

    @classmethod
    def normalize(cls, num):
        return (num - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, num):
        return num * cls.STD + cls.MEAN

    def resample_acc(self, index, split="val"):
        # when val_acc or test_acc are out of range
        assert split in ["val", "test"]
        split = 2 if split == "val" else 3
        for seed in range(3):
            acc = self.metrics[index, self.EPOCH_SET, seed, -1, split]
            #if not self._is_acc_blow(acc):
            return acc
        if self.debug:
            print(index, self.metrics[index, self.EPOCH_SET, :, -1])
            raise ValueError
        return np.array(self.MEAN)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    def _is_class_high_threshold(self, acc):
        return acc >= ACC_THRESHOLD #0.91


    def __getitem__(self, index):
        index = self.sample_range[index]

        self.seed = np.random.randint(0, 3)
        val_acc, test_acc = self.metrics[index, self.EPOCH_SET, self.seed, -1, 2:]
        val_avg = 0 
        test_avg = 0
        
        for seed_test_avg in [0, 1, 2]:
            val_acc, test_acc = self.metrics[index, self.EPOCH_SET, seed_test_avg, -1, 2:]
            val_avg += val_acc
            test_avg += test_acc
        
        val_acc = val_avg / 3
        test_acc = test_avg / 3

        n = self.num_vertices[index]
        ops_onehot = np.array([[i == k + 2 for i in range(5)] for k in self.operations[index]], dtype=np.float32)
        if n < 7:
            ops_onehot[n:] = 0.
        result = {
            "index": index,
            "num_vertices": n,
            "adjacency": self.adjacency[index],
            "operations": ops_onehot,
            "mask": np.array([i < n for i in range(7)], dtype=np.float32),
            "val_acc": val_acc,
            "test_acc": test_acc
        }
        if self.debug:
            self._check(result)
        return result

