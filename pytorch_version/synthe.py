import pypinyin
import difflib
import numpy as np
import os
import torch
import mfcc_reader
import constants as c
from veri_model_mfcc import Cnn_model
import torch.nn.functional as F
from numba import cuda

from ASRT_v0_6_1.SpeechModel251 import ModelSpeech

def Voiceprint_lock(real_wav_path,model_load_path,out_channel):

    wav_fa_path = "F:/文件/Tencent Files/904292841/FileRecv/904292841/FileRecv/真实语音数据/"
    pt_fa_path = "F:/文件/vox_data_pre/real/test/"
    finm_load_path = "F:/文件/Voiceprint_recognition_rewrite-master/a_real/file_name.npy"
    psw_load_path = "F:/文件/Voiceprint_recognition_rewrite-master/a_real/password.npy"
    fold_list = os.listdir(wav_fa_path)
    fold_list.sort()

    device = torch.device("cuda")

    class_name = np.load(finm_load_path)
    password = np.load(psw_load_path)

    thresh_1 = 0.73
    thresh_2 = 0.30

    datapath = 'ASRT_v0_6_1/'
    modelpath = 'ASRT_v0_6_1/model_speech/'

    ms = ModelSpeech(datapath)

    ms.LoadModel(modelpath + 'speech_model251_e_0_step_625000.model')

    real = ms.RecognizeSpeech_FromFile(real_wav_path)

    possible_list = []

    for id, psw in enumerate(password):
        posi = pypinyin.lazy_pinyin(psw)
        max_score = []
        for i in range(len(posi)):
            if i < len(real):
                score1 = difflib.SequenceMatcher(None,posi[i],real[i][:-1]).quick_ratio()
                if i - 1 >= 0:
                    score2 = difflib.SequenceMatcher(None,posi[i],real[i - 1][:-1]).quick_ratio()
                    if i + 1 < len(real):
                        score3 = difflib.SequenceMatcher(None,posi[i],real[i + 1][:-1]).quick_ratio()
                    elif i - 2 >= 0:
                        score3 = difflib.SequenceMatcher(None,posi[i],real[i - 2][:-1]).quick_ratio()
                    else:
                        score3 = 0
                elif i + 1 < len(real):
                    score2 = difflib.SequenceMatcher(None,posi[i],real[i + 1][:-1]).quick_ratio()
                    if i + 2 < len(real):
                        score3 = difflib.SequenceMatcher(None,posi[i],real[i + 2][:-1]).quick_ratio()
                    else:
                        score3 = 0
                else:
                    score2 = 0
                    score3 = 0

            elif i - 1 < len(real):
                if i - 1 >= 0:
                    score1 = difflib.SequenceMatcher(None,posi[i], real[i - 1][:-1]).quick_ratio()
                    if i - 2 >= 0:
                        score2 = difflib.SequenceMatcher(None,posi[i], real[i - 2][:-1]).quick_ratio()
                        if i - 3 >= 0:
                            score3 = difflib.SequenceMatcher(None,posi[i], real[i - 3][:-1]).quick_ratio()
                        else:
                            score3 = 0
                    else:
                        score2 = 0
                        score3 = 0
                else:
                    score1 = 0
                    score2 = 0
                    score3 = 0
            elif i - 2 < len(real):
                if i - 2 >= 0:
                    score1 = difflib.SequenceMatcher(None,posi[i], real[i - 2][:-1]).quick_ratio()
                    if i - 3 >= 0:
                        score2 = difflib.SequenceMatcher(None,posi[i], real[i - 3][:-1]).quick_ratio()
                        if i - 4 >= 0:
                            score3 = difflib.SequenceMatcher(None,posi[i], real[i - 4][:-1]).quick_ratio()
                        else:
                            score3 = 0
                    else:
                        score2 = 0
                        score3 = 0
                else:
                    score1 = 0
                    score2 = 0
                    score3 = 0
            elif i - 3 < len(real):
                if i - 3 >= 0:
                    score1 = difflib.SequenceMatcher(None,posi[i], real[i - 3][:-1]).quick_ratio()
                    if i - 4 >= 0:
                        score2 = difflib.SequenceMatcher(None,posi[i], real[i - 4][:-1]).quick_ratio()
                        if i - 5 >= 0:
                            score3 = difflib.SequenceMatcher(None,posi[i], real[i - 5][:-1]).quick_ratio()
                        else:
                            score3 = 0
                    else:
                        score2 = 0
                        score3 = 0
                else:
                    score1 = 0
                    score2 = 0
                    score3 = 0
            else:
                score1 = 0
                score2 = 0
                score3 = 0

            max_score.append(max(score1,score2,score3))

        score = np.array(max_score).mean()

        if score > thresh_1:
            possible_list.append(class_name[id][0:2])

    cuda.select_device(0)
    cuda.close()

    model = Cnn_model()

    if len(possible_list) > 0:
        model = torch.load(model_load_path).to(device)

    model.eval()

    veri_list = []

    for class_index in possible_list:
        inp0 = torch.load(pt_fa_path + str(class_index) + "_10.pt").unsqueeze(0).unsqueeze(0).float()
        mfcc = mfcc_reader.WavtoMfcc(real_wav_path,mode="test")
        inp1 = mfcc.readwav().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output0 = model(inp0.to(device),1,inp0.size(-2)).to(device)
            output1 = model(inp1.to(device),1,inp1.size(-2)).to(device)
        output = torch.zeros([2, 1, out_channel])
        output[0] = output0
        output[1] = output1

        dis = calculate_distance(output)

        if dis < thresh_2:
            veri_list.append([class_index,dis])

    if len(veri_list) > 0:
        veri_list = sorted(veri_list,key=lambda x:x[1])
        return veri_list
    else:
        return None

def calculate_distance(emb):
    distance = F.pairwise_distance(emb[0], emb[1], p=2) / 15
    return distance

if __name__ == '__main__':
    veri_list = Voiceprint_lock(c.REAL_WAV_PATH,c.VERI_MODEL_LOAD_PATH,c.OUT_CHANNEL)
    if veri_list:
        print(veri_list[0][0],",welcome!")
    else:
        print("音频未注册或密码不正确！")