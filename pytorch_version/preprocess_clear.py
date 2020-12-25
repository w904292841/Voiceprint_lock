import csv
import os
import os.path
import torch
import mfcc_reader

father_path = "F:/文件/voice/train/"
sub_path = ""
write_path = "F:/文件/voice_pre/"

all_list = []
class_head = []
file_head = []
file_name = []

def exisit_equal(lis):
    for i in range(len(lis)-1):
        if i == lis[-1]:
            return True
    return False


real_i = 0

fold_list = os.listdir(father_path)

min = 8

for i in range(48):

    done = 0

    sub_path = fold_list[i]

    # father_path+sub_path # F:/Vox_data/vox1_dev_wav/wav/id10001
    id_path = father_path+sub_path
    filelist = os.listdir(id_path) # 获取 id10001 下的所有文件
    filelist.sort()

    class_head.append(len(all_list))

    real_i = 0
    mode = "train"
    for id, filename in enumerate(filelist):
        this_vox_path = id_path + "/" + filename

        this_vox_path += " "

        mfcc = mfcc_reader.WavtoMfcc(this_vox_path,13,mode)
        data = mfcc.readwav()

        if data is not None:
            for j in range(data.size(0)):
                this_data = data[j]
                torch.save(this_data,write_path + mode + "/" + sub_path + "_" + str(real_i) + ".pt")
                real_i += 1
                print(sub_path," ",real_i)




# print(len(all_list))
# print(class_head)
