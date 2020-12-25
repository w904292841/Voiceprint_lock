import mfcc_reader
import torch

# import mfcc_reader

Father_path = "H:/科研/vox_data/"
dev_path = "vox1_dev_wav/"
test_path = "vox1_test_wav/"
sub_path = "vox1_dev_wav/"
write_path = "H:/科研/mfcc_data_pre/"
splite_file = "../a_iden/iden_split.txt"

all_list = []
class_head = []
file_head = []
file_name = []


def exisit_equal(lis):
    for i in range(len(lis) - 1):
        if i == lis[-1]:
            return True
    return False

with open(splite_file) as f:
    strings = f.readlines()
    labellist = [string.split(" ")[0] for string in strings]
    namelist = [string.split(" ")[1] for string in strings]
    f.close()


real_i = 0
min = 8
id = ""
sign = False
father_path = ""
old = ''
mode = ''
son_path_list = []

for id, l in enumerate(labellist):
    if l == '1' and int(namelist[id][2:7]) <= 10228:
        mode = 'train'
    elif l == '2' and int(namelist[id][2:7]) <= 10228:
        mode = 'dev'
    elif int(namelist[id][2:7]) <= 10228:
        mode = 'test'
    else:
        continue
    if namelist[id][:7] != old:
        old = namelist[id][:7]
        real_i = 0
        son_path_list = []
    file_path = Father_path + sub_path + namelist[id]
    son_path = namelist[id].split("/")[1]
    if son_path not in son_path_list:
        son_path_list.append(son_path)

    mfcc = mfcc_reader.WavtoMfcc(file_path, 16, mode)
    data = mfcc.readwav()

    for j in range(data.size(0)):
        this_data = data[j]
        real_i += 1
        torch.save(this_data, write_path + mode + "_veri" + "/" + namelist[id][:7] + "_" + str(son_path_list.index(son_path)) + "_" + str(real_i) + ".pt")
        print(namelist[id][:7], " ", real_i)

# print(len(all_list))
# print(class_head)
