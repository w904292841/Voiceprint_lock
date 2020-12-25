# coding=utf-8
from python_speech_features import mfcc
from python_speech_features import delta
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import torch
import constants as c


class WavtoMfcc(object):

    def __init__(self, ul, numc=13,mode="train"):

        self.numc = numc
        self.ul = ul.split("\n")[0]
        self.mode = mode

    def get_mfcc(self, data, fs):

        wav_feature = mfcc(data, fs, numcep=self.numc, winlen=c.FRAME_LEN, winstep=c.FRAME_STEP,
                           nfilt=26, nfft=c.NUM_FFT)
        #print(wav_feature.shape,"  before",type(wav_feature))

        reserve_length = wav_feature.shape[0] - wav_feature.shape[0]%100
        d_wav_feature_1 = delta(wav_feature, 2)
        d_wav_feature_2 = delta(d_wav_feature_1, 2)
        mfcc_feat_normal = normalize_frames(wav_feature.T)
        d_mfcc_feat_1_normal = normalize_frames(d_wav_feature_1.T)
        d_mfcc_feat_2_normal = normalize_frames(d_wav_feature_2.T)
        mfcc_feature = [mfcc_feat_normal, d_mfcc_feat_1_normal, d_mfcc_feat_2_normal]
        mfcc_feature = torch.tensor(mfcc_feature)
        length = (reserve_length / 100 - 1) / 2
        total_length = (int(length) * 2 + 1) * 2
        index = torch.randperm(total_length)
        if self.mode == "train":
            feature = torch.zeros([int(length),3,self.numc,300])
            for r in range(int(length)):
                for i in range(6):
                    feature[r, :, :, i * 50:(i + 1) * 50] =\
                        mfcc_feature[:, :, index[r * 4 + i] * 50:(index[r * 4 + i] + 1) * 50]
            feature = feature.permute(0,1,3,2)
        else:
            feature = mfcc_feature[:, :, 0:reserve_length]
            feature = feature.unsqueeze(0)
            feature = feature.permute(0,1,3,2)
        return feature

    def readwav(self):
        sample_rate, signal = scipy.io.wavfile.read(self.ul)
        mfcck = self.get_mfcc(signal, sample_rate)
        #print(mfcck.shape)
        # mfcck = mfcck[:,:,:300]
        #print(mfcck.shape)
        return mfcck


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])

if __name__ == '__main__':

    ul = 'F:/文件/vox_data/vox1_dev_wav/id10001/1zcIwhmdeo4/00001.wav'
    mfccdim = 13  # mfcc基本维数
    getmfcc = WavtoMfcc(ul, mfccdim,mode="train")
    mfcc= getmfcc.readwav()
    mfcc = torch.load('F:/文件/vox_data_pre/dev/id10001_9.pt')
    print(mfcc.shape)


