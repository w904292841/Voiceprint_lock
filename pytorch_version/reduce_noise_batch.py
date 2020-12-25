import os
from reduce_noise import reduce_noise1

father_path = "F:/文件/vox_data/"
sub_path = ""
father_write_path = "F:/文件/vox_data_noise_reduction/"

all_list = []
class_head = []
file_head = []
file_name = []

for i in range(1,129):
    if i > 269 and i<310:
        mode = "vox1_test_wav"
    else:
        mode = "vox1_dev_wav"

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
    id_path = father_path + mode + "/" + sub_path
    filelist = os.listdir(id_path) # 获取 id10001 下的所有文件
    filelist.sort()

    write_path = father_write_path + mode + "/" + sub_path
    os.mkdir(write_path)

    class_head.append(len(all_list))

    real_i = 0
    for id, filename in enumerate(filelist):

        all_vox_path = id_path + "/" + filename
        all_vox_list = os.listdir(all_vox_path)

        real_write_path = write_path + "/" + filename
        os.mkdir(real_write_path)

        for ID, vox_filename in enumerate(all_vox_list):
            real_i += 1
            this_vox_path = all_vox_path + "/" + vox_filename
            actully_write_path = real_write_path + "/" + vox_filename
            reduce_noise1(this_vox_path,actully_write_path)
            print(sub_path," ", real_i)