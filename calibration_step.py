# from torch.optim import lr_scheduler
# from EarlyStopping import EarlyStopping
import gc
import time
from torch import nn
from torch.optim import lr_scheduler

from Dataset_Final_CV_valdition import DatasetTest
import torch.utils.data as da
import numpy as np
import torch
from BaseModel_BN_ATT_DRT1111_bias5_multihead_self_liner import ConfuseNet
from Random_seed import seed_all
from Augmentation_GPU_test_2 import augmentation_2


def TwinningNet_train():
    # print('\nEpoch: %d' % epoch)
    # train_loss = 0
    # correct = 0
    # total = 0
    too_ = 0
    corr_ = 0
    test_batch = 32
    layers_with_dropout = [layer for layer in Dec.children() if isinstance(layer, nn.Dropout)]
    iter_test = iter(test_loader)
    # aug_samples = 0
    # addition_samples = 0
    # aug_labels = 0
    # loop through test data stream one by one
    scheduler_r = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(test_loader))
    for i in range(len(test_loader)):
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).cpu()

        # accumulate test data
        if i == 0:
            data_cum = inputs.float().cpu()
            # labels_cum = labels.float().cpu()
        elif 0 < i+1 <= test_batch:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)
            # labels_cum = torch.cat((labels_cum, labels.float().cpu()), 0)
        else:
            data_cum = torch.cat((data_cum[1:], inputs.float().cpu()), 0)

        inputs = data_cum[-1].numpy()
        inputs = inputs.reshape(-1, 1, inputs.shape[1], inputs.shape[2])

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

        Dec.eval()
        outputs_pre, _ = Dec(inputs)
        _, predicted_pre = outputs_pre.max(1)
        too_ += labels.size(0)
        corr_ += predicted_pre.eq(labels).sum().item()
        # softmax_p = outputs_pre.softmax(1)
        # p, predicted_label = softmax_p.max(1)

        Dec.train()
        # for m in Dec.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        #         # m.requires_grad_(False)
        #         # # force use of batch stats in train and eval modes
        #         # m.track_running_stats = False
        #         # m.running_mean = None
        #         # m.running_var = None
        
        # dynamic batch
        if (i + 1) >= test_batch:

            inputs = data_cum.numpy()
            inputs = inputs.reshape(test_batch, 1, inputs.shape[2], inputs.shape[3])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            p = 0
            with torch.no_grad():
                for U in range(30):
                    for layer in layers_with_dropout:
                        layer.p = torch.rand(1)*0.9
                    U_pre, _ = Dec(inputs)
                    U_softmax_p = U_pre.softmax(1)
                    if U == 0:
                        p = U_softmax_p
                    else:
                        p = torch.cat([p, U_softmax_p])
                p = p.reshape([-1, test_batch, 2])
                p_mean, p_var = p.mean(0), p.var(0)
                # p_var = p_var.mean(-1)
                pro, predicted_label = p_mean.max(1)
            tensor_isin = torch.where(pro >= 0.8)[0]
            # filter_ids_2 = torch.where(p_var < 0.001)[0]
            # tensor_isin = torch.isin(filter_ids_1, filter_ids_2)
            inputs = inputs[tensor_isin]
            predicted_label = predicted_label[tensor_isin]

            # with torch.no_grad():
            #     U_pre, _ = Dec(inputs)
            #     _, predicted_label = U_pre.max(1)

            # class_0 = torch.where(predicted_label == 0)[0]
            # class_1 = torch.where(predicted_label == 1)[0]
            # more_less = len(class_0) - len(class_1)
            # if len(class_0) and len(class_1) != 0:
            #     if more_less > 0:
            #         if more_less <= len(class_1):
            #             aug_samples, _ = augmentation_2(inputs[class_1][:more_less], repeat=1)
            #         else:
            #             # aug_num = more_less - len(class_1)
            #             aug_times = more_less // len(class_1)
            #             addition = more_less % len(class_1)
            #             aug_samples, _ = augmentation_2(inputs[class_1], repeat=aug_times)
            #             if addition != 0:
            #                 addition_samples, _ = augmentation_2(inputs[class_1][:addition], repeat=1)
            #                 aug_samples = torch.cat([aug_samples, addition_samples])
            #         aug_labels = torch.LongTensor(np.ones([len(aug_samples)])).cuda()
            #         inputs = torch.cat([inputs, aug_samples])
            #         predicted_label = torch.cat([predicted_label, aug_labels])
            #
            #     if more_less < 0:
            #         more_less = abs(more_less)
            #         if more_less <= len(class_0):
            #             aug_samples, _ = augmentation_2(inputs[class_0][:more_less], repeat=1)
            #         else:
            #             # aug_num = more_less - len(class_0)
            #             aug_times = more_less // len(class_0)
            #             addition = more_less % len(class_0)
            #             aug_samples, _ = augmentation_2(inputs[class_0], repeat=aug_times)
            #             if addition != 0:
            #                 addition_samples, _ = augmentation_2(inputs[class_0][:addition], repeat=1)
            #                 aug_samples = torch.cat([aug_samples, addition_samples])
            #         aug_labels = torch.LongTensor(np.zeros([len(aug_samples)])).cuda()
            #         inputs = torch.cat([inputs, aug_samples])
            #         predicted_label = torch.cat([predicted_label, aug_labels])

            # if p >= 0.7 and p_var[predicted_label] <= 0.01:
            optimizer.zero_grad()
            outputs, _ = Dec(inputs)
            loss = criterion(outputs, predicted_label)
            # loss_r = 0
            # for parameter in Dec.parameters():
            #     loss_r += torch.sum(parameter ** 2)
            # loss = criterion(outputs, predicted_label) + 0.0001 * loss_r
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(Dec.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler_r.step()

            # train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += labels.size(0)
            # correct += predicted.eq(labels).sum().item()

        # print(batch_idx, len(test_loader), 'trainLoss: %.3f | testAcc: %.3f%% (%d/%d)'
        #       % (train_loss / (batch_idx + 1), 100. * corr_ / too_, corr_, too_))

    # scheduler.step(loss)

    final_ada_acc = round(corr_ / too_, 4)
    # tr_loss = train_loss / (batch_idx + 1)

    return final_ada_acc


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    start_time = time.perf_counter()
    sub = 1
    Final_result_raw = []
    seed_all(1)

    for subject in range(sub, 53):
        gc.collect()
        torch.cuda.empty_cache()
        # print('***************Current subject: %d ***************' % subject)

        DatasetTest_set = DatasetTest(subject)
        test_feature = np.concatenate([DatasetTest_set['left'], DatasetTest_set['right']])
        test_label = np.concatenate([DatasetTest_set['left_label'], DatasetTest_set['right_label']]).astype('int')

        # feed test data into data_loader
        test_label = torch.LongTensor(test_label.flatten()).to(device)
        test_feature = torch.tensor(test_feature.swapaxes(1, 2))

        test_feature = torch.unsqueeze(test_feature, dim=1).type('torch.FloatTensor').to(device)
        test_data = da.TensorDataset(test_feature, test_label)
        test_loader = da.DataLoader(dataset=test_data, batch_size=1, shuffle=True, drop_last=False)

        # train
        Dec = ConfuseNet(num_classes=2)
        Dec.cuda()
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(Dec.parameters(), lr=0.00003, weight_decay=0.001)  # , weight_decay=0.001
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, cooldown=0, min_lr=0,
        #                                            verbose=False)
        # scaler = GradScaler()

        metric = 'loss'
        path_m = './ModelParam/GIST16_42/multi1_self_GZZ_ModelParameter%d.pt' % subject
        # path_m = './ModelParam/MMI16_42/bias5_multi1_self_liner_16_ModelParameter%d.pt' % subject
        # path_m = './ModelParam/OpenBMI16_42/bias5_multi1_self_liner_16_ModelParameter%d.pt' % subject
        # path_o = './model_param/GIST/bias5_multi1_self_OptimizerParameter%d.pt' % subject
        # early_stopping = EarlyStopping(11, metric=metric, path_m=path_m, path_o=path_o)
        Dec.load_state_dict(torch.load(path_m))
        # optimizer.load_state_dict(torch.load(path_o))

        file_name = './AdaResult/GIST_ConfT_AUG_woCB.npy'

        # 自适应测试开始
        test_acc_ada = TwinningNet_train()

        if subject == 1:
            Final_result_raw.append(test_acc_ada)
            np.save(file_name, Final_result_raw)
        else:
            Final_result_raw = np.load(file_name)
            Final_result_raw = list(Final_result_raw)
            Final_result_raw.append(test_acc_ada)
            np.save(file_name, Final_result_raw)
        print('Current subject:', subject, '    Ada_Acc:', test_acc_ada)

    current_time = time.perf_counter()
    running_time = current_time - start_time
    print('Acc:', np.asarray(Final_result_raw).mean(), 'Std', np.asarray(Final_result_raw).std())
    print("Total Running Time: {} seconds".format(round(running_time, 2)))

