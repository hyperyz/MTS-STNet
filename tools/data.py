import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


def get_dataloader(data_file, batch_size, device):
    data = load_original_data(data_file=data_file)
    data, scaler = normalize_data(data)

    steps_per_day = 288
    add_time_of_day = True
    add_day_of_week = True
    l, n, _ = data.shape
    feature_list = [data]
    if add_time_of_day:
        time_of_day = np.array([i % steps_per_day / steps_per_day for i in range(l)])
        time_of_day_tiled = np.tile(time_of_day, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day_tiled)

    if add_day_of_week:
        day_of_week = np.array([(i // steps_per_day) % 7 / 7 for i in range(l)])
        day_of_week_tiled = np.tile(day_of_week, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)
    data = np.concatenate(feature_list, axis=-1)  # L x N x C

    data_train, data_val, data_test = split_data_by_ratio(data, 0.2, 0.2)
    x_tra, y_tra = create_sliding_window(data_train, 12, 12, single=False)
    x_val, y_val = create_sliding_window(data_val, 12, 12, single=False)
    x_test, y_test = create_sliding_window(data_test, 12, 12, single=False)
    y_tra = y_tra[:, :, :, 0:1]
    y_val = y_val[:, :, :, 0:1]
    y_test = y_test[:, :, :, 0:1]
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    train_dataloader = data_loader(x_tra, y_tra, batch_size, device, shuffle=True, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, batch_size, device, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, batch_size, device, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def load_original_data(data_file):
    data = np.load(data_file)['data']
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print("npz中的原始数据：", data.shape)
    return data


def normalize_data(data):
    mean = data.mean()
    std = data.std()
    scaler = StandardScaler(mean, std)
    data = scaler.transform(data)
    print('标准化数据集')
    return data, scaler


def create_sliding_window(data, window, horizon, single):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window + horizon - 1:index + window + horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window:index + window + horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def data_loader(X, Y, batch_size, device, shuffle=True, drop_last=True):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    data = TensorDataset(X, Y)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


if __name__ == '__main__':
    import h5py

    with h5py.File('../data/pemsbay/PEMS-BAY.h5', 'r') as file:
        print("Keys:", list(file.keys()))

        speed_dataset = file['speed']

        # 将数据集转换为 numpy 数组
        speed_data = np.array(speed_dataset)

        # 打印 numpy 数组内容
        print(speed_data)
        print(type(speed_data))
        print(speed_data.shape)
