import csv
import os
import os.path
import numpy as np

father_path = "H:/科研/mfcc_data_pre/dev/"
write_path = "../a_iden/mfcc_test_list_pt_128.csv"

filelist = os.listdir(father_path)
filelist.sort()

with open(write_path,'a',newline='') as f:
    csv_writer = csv.writer(f)
    # 添加标题

    real_i = 0
    id = ""
    i = 0

    for this_vox_path in filelist:
        i = int(this_vox_path[3:7])
        if i > 128:
            break
        this_vox_path = this_vox_path.replace(father_path, "")
        csv_content = [this_vox_path,i-1]
        csv_writer.writerow(csv_content)
        real_i += 1
    print(real_i)

    f.close()