import torch
import tools
import constants as c

batch_size = 32


datalist,labellist = tools.get_voxceleb1_datalist(c.FA_DIR + "train/","../a_iden/mfcc_train_list_pt_128.txt")

fa_data = c.FA_DIR + "train_batch_32/"

data = tools.Loader_pt(datalist,labellist,c.MFCC_DIM,c.MAX_SEC,c.BUCKET_STEP,c.FRAME_STEP,batch_size,c.IDEN_CLASS)

for i in range(data.batchs()):
    data_batch, label_batch = data.batch_data_mfcc(i)
    all_data = (data_batch,label_batch)
    torch.save(all_data,fa_data + str(i) + ".pt")
    print("finish", i + 1, "/", data.batchs())
