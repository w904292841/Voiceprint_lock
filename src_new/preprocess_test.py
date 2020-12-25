import mfcc_reader
import csv
import os
import os.path
import torch

father_path = "/export/longfuhui/all_data/vox_data/"
dev_path = "vox1_dev_wav/"
test_path = "vox1_test_wav/"
splite_file = "../a_iden/iden_split.txt"
write_path = "/export/longfuhui/all_data/mfcc_data_pre/test/"
real_i = 0
old = ""

with open(splite_file) as f:
    strings = f.readlines()
    labellist = [string.split(" ")[0] for string in strings]
    namelist = [string.split(" ")[1] for string in strings]
    f.close()

for id, i in enumerate(labellist):
    if i == '3':
        if int(namelist[id][2:7]) > 269 and int(namelist[id][2:7]) < 310:
            sub_path = test_path
        else:
            sub_path = dev_path

        if namelist[id][2:7] != old:
            old = namelist[id][2:7]
            real_i = 0

        file_path = father_path + sub_path + namelist[id]
        mfcc = mfcc_reader.WavtoMfcc(file_path, 13, 'test')
        data = mfcc.readwav()
        real_i += 1
        torch.save(data,namelist[id][:7] + "_" + str(real_i) + ".pt")