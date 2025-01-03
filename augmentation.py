import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_torch(seed=20230206):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augmentation_2(samples, repeat=None, aug_=None, rand=False):
    """input: 16, 1, 16, 512"""
    if rand:
        seed_torch()

    if len(samples.shape) == 4:
        samples = torch.squeeze(samples, dim=1)

    if samples.shape[1] >= samples.shape[2]:
        samples = samples.swapaxes(1, 2)

    method_list = ['noise_adding', 'channel_fuse', 'delay', 'segment_stretch']
    aug_samples = []
    method_index_set = []

    if repeat >= 1:
        repeat = int(repeat)
        for i in range(repeat):
            for j in range(len(samples)):
                seg = samples[j]
                aug_seg = []
                random_seed = random.sample(range(0, 8), 1)[0]
                method = method_list[random_seed]
                if aug_ == 0:
                    method = 'noise_adding'
                if aug_ == 1:
                    method = 'channel_fuse'
                if aug_ == 2:
                    method = 'delay'
                if aug_ == 3:
                    method = 'segment_stretch'

                if method == 'noise_adding':
                    aug_seg = noise_adding(seg)
                    method_index = 0
                if method == 'channel_fuse':
                    aug_seg = channel_fuse(seg)
                    method_index = 1
                if method == 'delay':
                    aug_seg = delay(seg)
                    method_index = 2
                if method == 'segment_stretch':
                    aug_seg = segment_stretch(seg)
                    method_index = 3

                if j == 0 and i == 0:
                    aug_samples = aug_seg
                else:
                    aug_samples = torch.cat([aug_samples, aug_seg])
                method_index_set.append(method_index)
    else:
        aug_samples = samples

    aug_samples = aug_samples.reshape([-1, samples.shape[1], samples.shape[2]])
    aug_samples = torch.unsqueeze(aug_samples, dim=1)

    return aug_samples, method_index_set


def noise_adding(data):
    ch_num = random.sample(range(round(data.shape[0] / 2)), 1)[0]
    ch = np.sort(random.sample(range(data.shape[0]), ch_num))
    data_select_ch = data[ch]
    # sigma = random.uniform(0.001, 0.1)
    sample_noise = (0.01 ** 0.5) * torch.randn(ch.shape[0], data.shape[1]).to(device)
    aug_data_sum = data_select_ch + sample_noise
    # aug_data = data.copy()
    data[ch, :] = aug_data_sum

    return data


def delay(data):
    loc = random.sample(range(0, int(data.shape[1] / 2)), 1)[0]
    seg = data[:, loc:]
    # fill_up = (0.1 ** 0.5) * torch.randn(data.shape[0], data.shape[1] - seg.shape[1]).to(device)
    fill_up = torch.zeros(data.shape[0], data.shape[1] - seg.shape[1]).to(device)
    aug_data = torch.cat([fill_up, seg], dim=1)

    return aug_data


def segment_stretch(data):
    mode = random.sample(range(0, 2), 1)[0]
    if mode == 0:
        down_resample_list = [0.25, 0.5, 0.75]
        down_times = down_resample_list[random.sample(range(0, 3), 1)[0]]
        seg = F.interpolate(data.unsqueeze(0), scale_factor=down_times, mode='nearest',
                            recompute_scale_factor=True).squeeze(0)

        # fill_up = (0.1 ** 0.5) * torch.randn(data.shape[0], data.shape[1] - seg.shape[1]).to(device)
        fill_up = torch.zeros(data.shape[0], data.shape[1] - seg.shape[1]).to(device)
        loc = random.sample(range(0, data.shape[1] - seg.shape[1]), 1)[0]
        seg_half = torch.cat([fill_up[:, :loc], seg], dim=1)
        seg_final = torch.cat([seg_half, fill_up[:, loc:]], dim=1)
    else:
        up_resample_list = [1.25, 1.5, 1.75, 2]
        up_times = up_resample_list[random.sample(range(0, 4), 1)[0]]
        seg_final = F.interpolate(data.unsqueeze(0), scale_factor=up_times, mode='nearest',
                                  recompute_scale_factor=True).squeeze(0)[:, :512]

    return seg_final


def channel_fuse(data):
    left = {'f7': [0, 1, 4], 'f3': [0, 1, 5], 't7': [0, 4, 5, 9], 'c3': [1, 4, 5, 6, 10], 'p7': [4, 9, 10, 14],
            'p3': [5, 9, 10, 11, 14], 'o1': [9, 10, 14]}
    right = {'f8': [2, 3, 8], 'f4': [2, 3, 7], 't8': [3, 7, 8, 13], 'c4': [2, 6, 7, 8, 12], 'p8': [8, 12, 13, 15],
             'p4': [7, 11, 12, 13, 15], 'o2': [12, 13, 15]}

    electrode_all = np.arange(data.shape[0])
    # left
    electrode_random_select_left = list(left.keys())[random.sample(range(0, 7), 1)[0]]
    electrode_left = left[electrode_random_select_left]
    shuffle_electrode_left = electrode_left.copy()
    random.shuffle(shuffle_electrode_left)
    electrode_all[electrode_left] = shuffle_electrode_left
    # right
    electrode_random_select_right = list(right.keys())[random.sample(range(0, 7), 1)[0]]
    electrode_right = right[electrode_random_select_right]
    shuffle_electrode_right = electrode_right.copy()
    random.shuffle(shuffle_electrode_right)
    electrode_all[electrode_right] = shuffle_electrode_right

    aug_data = data.clone()
    aug_data = aug_data[electrode_all]

    return aug_data


if __name__ == '__main__':
    start_time = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k in range(100):
        dataset = torch.randn([32, 1, 16, 512]).to(device)
        label = torch.zeros([32, 1]).to(device)
        dataset_noise = augmentation_2(dataset, repeat=1, aug_=6)
        # print(dataset_noise.shape)
    current_time = time.perf_counter()
    running_time = current_time - start_time
    print("Total Running Time: {} seconds".format(round(running_time, 2)))

    # dataset = torch.randn([32, 16, 512]).to(device)
    # label = torch.zeros([32, 1]).to(device)
    # dataset_ture = torch.randn([32, 16, 512]).to(device)
    # dataset_ture_label = torch.zeros([32, 1]).to(device)
    #
    # dataset_noise, dataset_label = augmentation(dataset, label, repeat=1.0, fuse_ture=True, ture_data=dataset_ture, ture_label=dataset_ture_label, rand=False)

