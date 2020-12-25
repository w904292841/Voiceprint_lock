import csv
import os
import os.path
from random import randint

father_path = "F:/文件/vox_data_pre/real/test/"
write_path = "../a_veri/real_test_veri_list_pt.csv"

id = ""

filelist = os.listdir(father_path)

start = 0
end = len(filelist)

filelist.sort()

def exisit_equal(lis):
    for i in range(len(lis)-1):
        if i == lis[-1]:
            return True
    return False

with open(write_path,'a',newline='') as f:
    csv_writer = csv.writer(f)
    # 添加标题

    real_i = 0

    for i in range(len(filelist)):

        if filelist[i][3:5] == "10":
            nega_ram = []
            for j in range(3):
                vox_path = filelist[i] + " " + filelist[i-j-1]

                csv_content = [1, vox_path]
                csv_writer.writerow(csv_content)
                real_i += 1

                nega_ram.append(randint(start,end - 1))
                while filelist[nega_ram[j]][0:2] == filelist[i][0:2] or filelist[nega_ram[j]][3:5] == "10" or exisit_equal(nega_ram):
                    nega_ram[j] = randint(start,end - 1)

                vox_path = filelist[i] + " " + filelist[nega_ram[j]]

                csv_content = [0, vox_path]
                csv_writer.writerow(csv_content)
                real_i += 1
    print(real_i)

    f.close()