import csv
import os
import os.path
from random import randint

father_path = "F:/文件/vox_data_pre/train/"
write_path = "../a_veri/train_veri_list_new.csv"

class_head = []
id = ""

filelist = os.listdir(father_path)

def exisit_equal(lis):
    for i in range(len(lis)-1):
        if i == lis[-1]:
            return True
    return False

for ID, file in enumerate(filelist):
    if file[:7] != id:
        id = file[:7]
        class_head.append(ID)
class_head.append(len(filelist))

with open(write_path,'a',newline='') as f:
    csv_writer = csv.writer(f)
    # 添加标题

    real_i = 0

    for i in range(1, 257):

        classlist = filelist[class_head[i-1]:class_head[i]]
        length = class_head[i] - class_head[i - 1]
        x = 0

        for ID, this_vox_path in enumerate(classlist):
            while this_vox_path[8] != str(x) and this_vox_path[8:10] != str(x):
                x += 1

            nega_ram = []
            posi_ram = []

            for j in range(3):

                posi_ram.append(randint(class_head[i - 1], class_head[i] - 1))
                while filelist[posi_ram[j]][8] == str(x) or filelist[posi_ram[j]][8:10] == str(x) or exisit_equal(
                        posi_ram):
                    posi_ram[j] = randint(class_head[i - 1], class_head[i] - 1)

                nega_ram.append(randint(0, len(filelist) - 1))
                while nega_ram[j] >= class_head[i - 1] and nega_ram[j] < class_head[i] or exisit_equal(nega_ram):
                    nega_ram[j] = randint(0, len(filelist) - 1)

                vox_path = this_vox_path
                vox_path += " "
                vox_path += filelist[posi_ram[j]]
                vox_path += " "
                vox_path += filelist[nega_ram[j]]

                csv_content = [vox_path]
                csv_writer.writerow(csv_content)
                real_i += 1
    print(real_i)

    f.close()