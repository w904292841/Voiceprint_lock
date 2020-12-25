# coding=utf-8
import os
os.environ["MKL_NUM_THREADS"] = '12'
os.environ["NUMEXPR_NUM_THREADS"] = '12'
os.environ["OMP_NUM_THREADS"] = '12'
import numpy as np
import torch
import constants as c
# from wav_reader import get_fft_spectrum
# from mfcc_reader import WavtoMfcc
import matplotlib.pyplot as plt
import os

'''
设置多线程
os.environ["MKL_NUM_THREADS"] = '12'
os.environ["NUMEXPR_NUM_THREADS"] = '12'
os.environ["OMP_NUM_THREADS"] = '12'
'''

'''
读入训练文件列表 
FA_DIR：音频文件绝对路径前缀
path：训练文件列表
'''


def get_voxceleb1_datalist(FA_DIR, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = [os.path.join(FA_DIR, string.split(",")[0]) for string in strings]
        labellist = [int(string.split(",")[1]) for string in strings]
        # print(len(labellist))
        f.close()
        # audiolist = audiolist.flatten()
        # labellist = labellist.flatten()
        # print(audiolist.shape)
        # print(labellist.shape)
    return audiolist, labellist

def get_veri_datalist(FA_DIR, path):
    with open(path) as f:
        strings = f.readlines()
        verilist = [[FA_DIR + string.split(" ")[0], FA_DIR + string.split(" ")[1],
                     FA_DIR + string.split(" ")[2].split("\n")[0]] for string in strings]
        # print(len(labellist))
        f.close()
        # audiolist = audiolist.flatten()
        # labellist = labellist.flatten()
        # print(audiolist.shape)
        # print(labellist.shape)
    return verilist

def get_veri_test_datalist(FA_DIR, path):
    with open(path) as f:
        strings = f.readlines()
        labellist = [int(string.split(",")[0]) for string in strings]
        audiolist = [string.split(",")[1] for string in strings]
        verilist = [[FA_DIR + string.split(" ")[0],FA_DIR + string.split(" ")[1].split("\n")[0]]
                    for string in audiolist]
        # print(len(labellist))
        f.close()
        # audiolist = audiolist.flatten()
        # labellist = labellist.flatten()
        # print(audiolist.shape)
        # print(labellist.shape)
    return verilist, labellist


'''
方法：计算不同帧数对应的最终输出尺寸是否大于 0
'''


def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)  # 1s/10ms = 100
    end_frame = int(max_sec * frames_per_sec)  # 10s*100 = 1000
    step_frame = int(step_sec * frames_per_sec)  # 1*100 = 100
    for i in range(0, end_frame + 1, step_frame):  # [100,...,1000]
        s = i  # 100,200,300,...,1000
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1  np.floor()返回不大于输入参数的最大整数,向下取整
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


'''
类：训练数据的生成器
'''


class Loader():
    def __init__(self,list_IDs, labels, dim, max_sec, step_sec, frame_step, batch_size=2, n_classes=1251,
                 shuffle=True,mode="train"):
        self.list_IDs = list_IDs
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_sec = max_sec
        self.step_sec = step_sec
        self.frame_step = frame_step
        self.buckets = build_buckets(self.max_sec, self.step_sec, self.frame_step)
        self.on_epoch_end()
        self.mode = mode

    def on_epoch_end(self):
        '每次迭代后打乱训练列表'
        self.indexes = torch.arange(len(self.list_IDs))
        if self.shuffle:
            self.indexes = torch.randperm(self.indexes.size(0))

    def batch_data(self,index):
        '返回一个 batch 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_data, batch_labels = self.load_data(list_IDs_temp, indexes)
        return batch_data, batch_labels

    def batch_data_mfcc(self,index):
        '返回一个 batch 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_data, batch_labels = self.load_mfccdata(list_IDs_temp, indexes)
        return batch_data, batch_labels

    def batchs(self):
        '计算有多少个 batch'
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def load_data(self, list_IDs_temp, indexes):
        '得到频谱数组和类标签，以输入模型进行训练'
        b_data = torch.rand((self.batch_size,) + self.dim)
        b_labels = torch.ones((self.batch_size,))

        for i, ID in enumerate(list_IDs_temp):

            b_data[i, 0] = torch.tensor(get_fft_spectrum(ID,build_buckets(c.MAX_SEC, c.BUCKET_STEP,c.FRAME_STEP)).tolist())
            b_labels[i] = torch.tensor(self.labels[indexes[i]])  # 0~n-1
            # b_labels[i] = self.labels[indexes[i]] - 1 # 1~n

        # # 转换为独热码
        # b_labels = b_labels.view(self.batch_size,1)
        # b_labels = b_labels.repeat(1,self.n_classes)
        # for i in range(self.batch_size):
        #     b_labels[i] = torch.tensor([int(b_labels[i,j]==j) for j in range(self.n_classes)])

        return b_data, b_labels

    def load_mfccdata(self,list_IDs_temp, indexes):

        b_data = torch.rand((self.batch_size,) + self.dim)
        b_labels = torch.ones((self.batch_size,))

        for i, ID in enumerate(list_IDs_temp):

            mfccdim = 13  # mfcc基本维数
            getmfcc = WavtoMfcc(ID, mfccdim,self.mode)
            mfcc = getmfcc.readwav()
            b_data[i] = mfcc
            b_labels[i] = torch.tensor(self.labels[indexes[i]])  # 0~n-1

        return b_data, b_labels



class Loader_veri():
    def __init__(self,list_IDs, dim, max_sec, step_sec, frame_step, batch_size=2, n_classes=1251,labels=None,
                 shuffle=True,mode="train",index=None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_sec = max_sec
        self.step_sec = step_sec
        self.frame_step = frame_step
        self.buckets = build_buckets(self.max_sec, self.step_sec, self.frame_step)
        self.index = index
        self.on_epoch_end()
        self.mode = mode

    def on_epoch_end(self):
        '每次迭代后打乱训练列表'
        self.indexes = torch.arange(len(self.list_IDs))
        if self.index != None:
            self.indexes = self.index
        elif self.shuffle:
            self.indexes = torch.randperm(self.indexes.size(0))

    def batch_data(self,index):
        '返回一个 batch 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_data = self.load_data(list_IDs_temp, indexes)
        return batch_data

    def batch_data_mfcc(self,index):
        '返回一个 batch 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_data = self.load_mfccdata(list_IDs_temp, indexes)
        return batch_data

    def batch_data_mfcc_test(self,index):
        '返回一个 batch 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_data, batch_labels = self.load_mfccdata_test(list_IDs_temp, indexes)
        return batch_data, batch_labels

    def batchs(self):
        '计算有多少个 batch'
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def load_data(self, list_IDs_temp, indexes):
        '得到频谱数组和类标签，以输入模型进行训练'
        b_data = torch.rand((self.batch_size,3,) + self.dim)

        for i, ID in enumerate(list_IDs_temp):

            b_data[i, 0, 0] = torch.tensor(
                get_fft_spectrum(ID[0],build_buckets(c.MAX_SEC, c.BUCKET_STEP,c.FRAME_STEP)).tolist())
            b_data[i, 1, 0] = torch.tensor(
                get_fft_spectrum(ID[1], build_buckets(c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)).tolist())
            b_data[i, 2, 0] = torch.tensor(
                get_fft_spectrum(ID[2], build_buckets(c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)).tolist())
            # b_labels[i] = self.labels[indexes[i]] - 1 # 1~n

        # # 转换为独热码
        # b_labels = b_labels.view(self.batch_size,1)
        # b_labels = b_labels.repeat(1,self.n_classes)
        # for i in range(self.batch_size):
        #     b_labels[i] = torch.tensor([int(b_labels[i,j]==j) for j in range(self.n_classes)])

        return b_data

    def get_index(self):
        return self.indexes

    def load_mfccdata(self,list_IDs_temp, indexes):

        b_data = torch.rand((self.batch_size,3,) + self.dim)

        for i, ID in enumerate(list_IDs_temp):

            mfccdim = 13  # mfcc基本维数
            mfcc = torch.load(ID[0])
            b_data[i,0] = mfcc

            mfcc = torch.load(ID[1])
            b_data[i, 1] = mfcc

            mfcc = torch.load(ID[2])
            b_data[i, 2] = mfcc

        return b_data

    def load_mfccdata_test(self,list_IDs_temp, indexes):

        b_data = []
        b_labels = torch.ones((self.batch_size,))

        for i, ID in enumerate(list_IDs_temp):

            # print(ID[0])
            mfcc = torch.load(ID[0]).unsqueeze(0).unsqueeze(0).float()
            b_data.append(mfcc)
            # print(ID[1])
            mfcc = torch.load(ID[1]).unsqueeze(0).unsqueeze(0).float()
            b_data.append(mfcc)

            b_labels[i] = torch.tensor(self.labels[indexes[i]])  # 0~n-1

        return b_data, b_labels

class Loader_pt():
    def __init__(self,list_IDs, labels, dim, max_sec, step_sec, frame_step, batch_size=2, n_classes=1251,
                 shuffle=True,mode="train",index=None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_sec = max_sec
        self.step_sec = step_sec
        self.frame_step = frame_step
        self.buckets = build_buckets(self.max_sec, self.step_sec, self.frame_step)
        self.index = index
        self.on_epoch_end()
        self.mode = mode

    def on_epoch_end(self):
        '每次迭代后打乱训练列表'
        self.indexes = torch.arange(len(self.list_IDs))
        if self.index != None:
            self.indexes = self.index
        elif self.shuffle:
            self.indexes = torch.randperm(self.indexes.size(0))

    def batch_data(self,index):
        '返回一个 batch 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_data = self.load_data(list_IDs_temp, indexes)
        return batch_data

    def batch_data_mfcc(self,index):
        '返回一个 batch 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        if self.mode == "train":
            batch_data = self.load_mfccdata(list_IDs_temp, indexes)
        else:
            batch_data = self.load_mfccdata_test(list_IDs_temp, indexes)
        return batch_data

    def load_data(self, list_IDs_temp, indexes):
        b_data = torch.rand((self.batch_size,) + self.dim)
        b_labels = torch.ones((self.batch_size,))

        for i, ID in enumerate(list_IDs_temp):
            b_data[i] = torch.load(ID)
            b_labels[i] = torch.tensor(self.labels[indexes[i]])

        return b_data, b_labels

    def load_mfccdata(self, list_IDs_temp, indexes):
        b_data = torch.rand((self.batch_size,) + self.dim)
        b_labels = torch.ones((self.batch_size,))

        for i, ID in enumerate(list_IDs_temp):
            # print(ID)
            b_data[i] = torch.load(ID)
            b_labels[i] = torch.tensor(self.labels[indexes[i]])

        return b_data, b_labels

    def load_mfccdata_test(self, list_IDs_temp, indexes):
        # print(list_IDs_temp[0])
        b_data = torch.load(list_IDs_temp[0]).unsqueeze(0).float()
        b_labels = torch.tensor(self.labels[indexes[0]]).unsqueeze(0).float()

        return b_data, b_labels

    def batchs(self):
        '计算有多少个 batch'
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def get_index(self):
        return self.indexes

'''绘制训练损失图像'''


def draw_loss_img(history_dict,save_path):
    loss_values = history_dict['loss']  # 训练损失
    # val_loss_values = history_dict['val_loss']  # 验证损失
    ep = range(1, len(loss_values) + 1)

    plt.switch_backend('agg')
    plt.plot(ep, loss_values, 'b', label="Training loss")  # bo表示蓝色原点
    # plt.plot(ep, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


'''绘制训练精度图像'''


def draw_acc_img(history_dict,save_path):
    accs = history_dict['acc']  # 训练精度
    # val_acc = history_dict['val_acc']  # 验证精度
    ep = range(1, len(accs) + 1)

    plt.switch_backend('agg')
    plt.plot(ep, accs, 'b', label="Training Acc")  # bo表示蓝色原点
    # plt.plot(ep, val_acc, 'b', label="Validation Acc")  # b表示蓝色实线
    plt.title("Train Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()  # 绘图
    plt.savefig(save_path)
    plt.show()

def draw_roc(fpr,tpr,auc,model_name):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("roc_for_{}".format(model_name))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("../roc_img/","roc_for_{}.png".format(model_name)))
    # plt.show()

def draw_eer(fpr,tpr,thresholds,eer,thresh,model_name):

    plt.figure()
    plt.plot(thresholds,1 - tpr, marker='*', label='frr')
    plt.plot(thresholds,fpr, marker='o', label='far\n' + 'eer = %0.2f\n' % eer + 'thresh = %0.2f' % thresh)   # label='fpr\n' + 'eer = %0.2f\n' % eer + 'thresh = %0.2f' % thresh
    # lw = 2
    # plt.plot(lw=lw, label='eer = %0.2f\n' % eer + 'thresh = %0.2f' % thresh)
    # plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('thresh')
    plt.ylabel('frr/far')
    plt.title("eer_for_{}".format(model_name))
    plt.legend()
    plt.savefig(os.path.join("../eer_img/", "eer_for_{}.png".format(model_name)))
    # plt.show()


def calculate_eer(y, y_score,model_name):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn import metrics
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    model_name = model_name[0:-4]

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=0.)
    auc = metrics.auc(fpr, tpr)
    draw_roc(fpr,tpr,auc,model_name)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    draw_eer(fpr,tpr,thresholds,eer,thresh,model_name)
    # exit(0)

    return eer

def test():
    audiolist, labellist = get_voxceleb1_datalist(c.FA_DIR, c.IDEN_TRAIN_LIST_FILE)

if __name__ == "__main__":
    test()
