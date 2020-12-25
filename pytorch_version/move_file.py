import os
import os.path
import shutil

test_path = "F:/文件/vox_data_pre/wav/test/"
train_path = "F:/文件/vox_data_pre/wav/train/"

id = ""

num = 0

filelist = os.listdir(test_path)

for ID, file in enumerate(filelist):
    if file[:7] != id:
        id = file[:7]
        num = 0
    if num > 3:
        shutil.move(test_path + file, train_path)
    num += 1