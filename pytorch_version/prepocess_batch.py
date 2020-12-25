import torch
import tools
import constants as c

batch_size = 64

datalist= tools.get_veri_datalist(c.FA_DIR + "train_veri/",c.VERI_TRAIN_LIST_FILE)

fa_data = c.FA_DIR + "train_batch_veri_64/"

data = tools.Loader_veri(datalist,c.MFCC_DIM,c.MAX_SEC,c.BUCKET_STEP,c.FRAME_STEP,batch_size,c.IDEN_CLASS)

for i in range(data.batchs()):
    data_batch = data.batch_data_mfcc(i)
    torch.save(data_batch,fa_data + str(i) + ".pt")
    print("finish", i + 1, "/", data.batchs())