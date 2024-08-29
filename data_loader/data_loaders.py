from torchvision import datasets, transforms
from base import BaseDataLoader, CamusBaseDataLoader
from data_loader.data.config import Subset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
from data_loader.data.camus.dataset import Camus


class CamusDataLoader(CamusBaseDataLoader):
    """
    Camus data loading using BaseDataLoader
    """
    def __init__(self, data_dir,  batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,):
        
        self.dataset = Camus(path = data_dir,
        image_set = Subset.TRAIN,
        fold = 1,
        predict = not training,
        data_augmentation = "pixel",)        
        self.data_val = Camus(path = data_dir,
        image_set = Subset.VAL,
        fold = 1,
        predict = not training,
        data_augmentation = "pixel",)
        # print("len of data val")
        # print(len(self.data_val))
        
        self.data_test = Camus(path = data_dir,
        image_set = Subset.TEST,
        fold = 1,
        predict = not training,
        data_augmentation = "pixel",)
        
        super().__init__(self.dataset, self.data_val, self.data_test, batch_size, shuffle, validation_split, num_workers)
        
class Test_CamusDataLoader(CamusBaseDataLoader):
    """
    Camus data loading using BaseDataLoader
    """
    def __init__(self, data_dir,  batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,):
        
        self.dataset = Camus(path = data_dir,
        image_set = Subset.TEST,
        fold = 1,
        predict = not training,
        data_augmentation = "pixel",)
        
        super().__init__(self.dataset, self.dataset, self.dataset, batch_size, shuffle, validation_split, num_workers)