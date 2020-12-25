import torch
import os
import random
import csv

write_path = "../a_veri/train_veri_list_pt.csv"
load_path = "H:/科研/mfcc_data_pre/"
sub_path = "train_veri/"
file_path = load_path + sub_path

def exisit_equal(lis, aix):
    for i in range(len(lis)):
        if i == aix:
            return True
    return False

ori_list = os.listdir(file_path)

ori_list.sort()
# print(ori_list)

train_veri_list = []
ori_id = ""
ori_path = ""
for i in ori_list:
    recent_id = i.split("_")[0]
    recent_path = i.split("_")[1]
    if recent_id != ori_id:
        ori_id = recent_id
        train_veri_list.append([])
        if ori_id == "id10129":
            train_veri_list.pop()
            break
    if recent_path != ori_path:
        ori_path = recent_path
        train_veri_list[-1].append([])
    train_veri_list[-1][-1].append(i)

with open(write_path,mode="a",newline='') as f:
    csv_writer = csv.writer(f)
    for i, id in enumerate(train_veri_list):
        for j, path in enumerate(id):
            num = 0
            for m, file in enumerate(path):
                nega_id_list = []
                for k in range(3):
                    triple_list = file
                    next_path = (j + k + 1) % len(id)
                    pos_sample = id[next_path][num % len(id[next_path])]
                    nega_id = random.randint(0,len(train_veri_list) - 1)
                    while(exisit_equal(nega_id_list,nega_id)):
                        nega_id = random.randint(0,len(train_veri_list) - 1)
                    nega_id_list.append(nega_id)
                    nega_path = random.randint(0,len(train_veri_list[nega_id]) - 1)
                    nega_file = random.randint(0,len(train_veri_list[nega_id][nega_path]) - 1)
                    nega_sample = train_veri_list[nega_id][nega_path][nega_file]
                    triple_list += " " + pos_sample + " " + nega_sample
                    csv_writer.writerow([triple_list])
                    print(i + 1, "/", j + 1, "/", m + 1)