import os
import os.path
import errno
import numpy as np
import torch
import torch.utils.data as data
import codecs
from torch.utils.data import DataLoader



class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.use_cuda = torch.cuda.is_available()

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.train_data = self.train_data.view(self.train_data.size(0), -1).float()*0.02
            # self.train_data = self.train_data.view(self.train_data.size(0), -1).float()/255
            self.train_labels = self.train_labels.int()
            if self.use_cuda:
                self.train_data = self.train_data.cuda()
                self.train_labels = self.train_labels.cuda()
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data = self.test_data.view(self.test_data.size(0), -1).float()*0.02
            # self.test_data = self.test_data.view(self.test_data.size(0), -1).float()/255
            self.test_labels = self.test_labels.int()
            if self.use_cuda:
                self.test_data = self.test_data.cuda()
                self.test_labels = self.test_labels.cuda()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)
def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

import scipy.io as sio
class MNIST2000(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\M1000space1.mat')
        data_dict = dict(data_0)
        target = data_dict['trainLabels']  # 数据集的标签1*2000
        data = data_dict['trainImages_space']  # 数据2000*784
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 784)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class MNIST6000(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\minist6000.mat')
        data_dict = dict(data_0)
        target = data_dict['label_a']  # 数据集的标签1*2000
        data = data_dict['X']  # 数据6000*784
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 784)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class NUS01(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\nus01.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*30000
        data = data_dict['X2']  # 数据30000*144
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 144)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class MNIST01(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\mnist01.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*30000
        data = data_dict['X2']  # 数据30000*144
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 780)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class MNIST05(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\mnist05.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*30000
        data = data_dict['X']  # 数据30000*144
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 780)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class NUS101(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\nus-11.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*30000
        data = data_dict['X3']  # 数据30000*144
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 144)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class CCV01(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\ccv01.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*6773
        data = data_dict['X2']  # 数据6773*4000
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 4000)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class IAPR01(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\iapr01.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*6773
        data = data_dict['X2']  # 数据6773*4000
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 100)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class IAPR101(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\iapr-11.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*6773
        data = data_dict['X3']  # 数据6773*4000
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 100)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

class COIL101(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\coil20-11.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*6773
        data = data_dict['X3']  # 数据6773*4000
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 16384)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

class MNIST05(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\mnist05.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*30000
        data = data_dict['X']  # 数据30000*144
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 780)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class ESP01(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\esp01.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*6773
        data = data_dict['X2']  # 数据6773*4000
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 100)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class FLICK01(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\flickr01.mat')
        data_dict = dict(data_0)
        target = data_dict['label']  # 数据集的标签1*6773
        data = data_dict['X2']  # 数据6773*4000
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 100)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class USPS(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\USPS.mat')
        data_dict = dict(data_0)
        target = data_dict['Y']  # 数据集的标签1*1854
        data = data_dict['X']  # 数据1854*256
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 256)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class USPS_noise(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.3\\lib\\data\\USPS_noise.mat')
        data_dict = dict(data_0)
        target = data_dict['Y']  # 数据集的标签1*1854
        data = data_dict['Xocc']  # 数据1854*256
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 256)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class COIL20_noise(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.3\\lib\\data\\COIL20_noise.mat')
        data_dict = dict(data_0)
        target = data_dict['Y']  # 数据集的标签1*1440
        data = data_dict['Xocc']  # 数据1440*1024
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 1024)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class M05_view1(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\mnist2000_05_view1.mat')
        data_dict = dict(data_0)
        target = data_dict['groundtruth']  # 数据集的标签1*1440
        data = data_dict['X_view1']  # 数据1440*1024
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 784)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
class M05_view2(data.Dataset):
    def __init__(self):

        data_0 = sio.loadmat('G:\\experiment1.1\\lib\\data\\mnist2000_05_view2.mat')
        data_dict = dict(data_0)
        target = data_dict['groundtruth']  # 数据集的标签1*1440
        data = data_dict['X_view2']  # 数据1440*1024
        target_tensor = torch.from_numpy(target).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
        data_tensor = torch.from_numpy(data).to(torch.float32).view(-1, 784)  # 转化为张量
        self.data = data_tensor
        self.label = target_tensor

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
'''train_loder=MNIST2000()
# 读取数据
datas = DataLoader(train_loder, batch_size=200, shuffle=True, drop_last=False, num_workers=0)
for i, (data,target) in enumerate(datas):
    print("第 {} 个Batch \n{}".format(i, target))'''
