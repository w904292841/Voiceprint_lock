import torch
import numpy as np
import os

father_path = '/export/longfuhui/all_data/z_teacher_npy/mfcctrain/'
sub_path = ""
write_path = "/export/longfuhui/all_data/z_teacher_npy/pt_data/"

all_list = []
class_head = []
file_head = []
file_name = []
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
        mode = "train"
        # mode = "dev"

        all_vox_path = os.path.join(id_path, filename)
        all_vox_list = os.listdir(all_vox_path)

        for ID, vox_filename in enumerate(all_vox_list):
            this_vox_path = os.path.join(all_vox_path, vox_filename)
            this_vox_path += " "

            this_vox_path = this_vox_path.replace("\\", "/")

            mfcc = np.load(this_vox_path)
            data = torch.from_numpy(mfcc)
            if mode == 'dev':
                data = data.unsqueeze(0)

            for j in range(data.size(0)):
                this_data = data[j]
                torch.save(this_data,write_path + mode + "/" + sub_path + "_" + str(id) + "_" + str(real_i) + ".pt")
                real_i += 1
                print(sub_path," ",real_i)