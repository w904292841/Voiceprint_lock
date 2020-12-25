import csv
import os
import os.path
import torch
import mfcc_reader

father_path = "H:/科研/vox_data/vox1_dev_wav/"
sub_path = ""
write_path = "H:/科研/vox_data_pre/"

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

for i in range(1,257):

    done = 0

    if(i>= 1000):
        sub_path = "id1" + str(i)
    elif(i>=100):
        sub_path = "id10" + str(i)
    elif(i>=10):
        sub_path = "id100" + str(i)
    else:
        sub_path = "id1000" + str(i)
    # print(sub_path)

    # father_path+sub_path # F:/Vox_data/vox1_dev_wav/wav/id10001
    id_path = father_path+sub_path
    filelist = os.listdir(id_path) # 获取 id10001 下的所有文件
    filelist.sort()

    class_head.append(len(all_list))

    real_i = 0
    for id, filename in enumerate(filelist):
        if id < len(filelist) - 1:
            mode = "train"
        else:
            mode = "dev"

        all_vox_path = os.path.join(id_path, filename)
        all_vox_list = os.listdir(all_vox_path)

        for ID, vox_filename in enumerate(all_vox_list):
            this_vox_path = os.path.join(all_vox_path, vox_filename)
            this_vox_path += " "

            this_vox_path = this_vox_path.replace("\\", "/")

            mfcc = mfcc_reader.WavtoMfcc(this_vox_path,16,mode)
            data = mfcc.readwav()

            for j in range(data.size(0)):
                this_data = data[j]
                torch.save(this_data,write_path + mode + "/" + sub_path + "_" + str(id) + "_" + str(real_i) + ".pt")
                real_i += 1
                print(sub_path," ",real_i)




# print(len(all_list))
# print(class_head)
