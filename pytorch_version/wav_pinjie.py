# python 音频文件处理
# 多个 wav 格式语音文件合成
from pydub import AudioSegment
import os

father_path = "/export/longfuhui/all_data/vox_data/"
write_father_path = "/export/longfuhui/all_data/large_data/"
dev_path = "vox1_dev_wav/"
test_path = "vox1_test_wav/"
splite_file = "../a_iden/iden_split.txt"

with open(splite_file) as f:
    strings = f.readlines()
    labellist = [string.split(" ")[0] for string in strings]
    namelist = [string.split(" ")[1] for string in strings]
    f.close()

for i in range(1251):
    done = 0
    if i > 269 and i < 310:
        real_path = father_path + test_path
    else:
        real_path = father_path + dev_path

    if (i >= 1000):
        sub_path = "id1" + str(i)
    elif (i >= 100):
        sub_path = "id10" + str(i)
    elif (i >= 10):
        sub_path = "id100" + str(i)
    else:
        sub_path = "id1000" + str(i)

    id_path = real_path + sub_path + '/'
    write_path = write_father_path + real_path + sub_path
    filelist = os.listdir(id_path) # 获取 id10001 下的所有文件
    filelist.sort()
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    large_wav_file = AudioSegment.empty()
    for path in filelist:
        wav_file = id_path + '/' + path
        wav_path = os.listdir(wav_file)
        for wav in wav_path:
            recent_file = sub_path + '/' + path + '/' + wav
            if(labellist[namelist.index(recent_file)] == '3'):
                break
            sound = AudioSegment.from_file(wav, format="wav")
            large_wav_file += sound
    large_wav_file.export(write_path + 'all.wav',format="wav") #wav
