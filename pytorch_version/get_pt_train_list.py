import csv
import os
import os.path

father_path = "F:/文件/vox_data_noi_red_pre/train/"
write_path = "../a_iden/train_list_pt_noi_red.csv"

filelist = os.listdir(father_path)

with open(write_path,'a',newline='') as f:
    csv_writer = csv.writer(f)
    # 添加标题

    real_i = 0
    id = ""
    i = 0

    for this_vox_path in filelist:
        if id != this_vox_path[:7]:
            id = this_vox_path[:7]
            i += 1
            if i > 256:
                break
        this_vox_path = this_vox_path.replace(father_path, "")
        csv_content = [this_vox_path, i - 1]
        csv_writer.writerow(csv_content)
        real_i += 1

    print(real_i)
    f.close()