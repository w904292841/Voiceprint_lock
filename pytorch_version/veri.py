import os
import torch
import visdom  #(可视化工具包)下载：pip install visdom -i https://pypi.tuna.tsinghua.edu.cn/simple
from veri_model_mfcc import Cnn_model
import constants as c
import tools
from torch import optim
from torch import nn
from torch.nn import functional as F
from am_softmax import AMSoftmax

viz=visdom.Visdom(env = 'Cnn_mfcc_veri')  #运行程序前命令行输入： python -m visdom.server
device = torch.device('cuda')

def veri_vggvox_model(model_load_path, fa_data_dir, margin,batch_size, number,out_channel,
                      continue_training, lr, weight_decay, model_save_path,numc):

    veri_path = fa_data_dir + "train_batch_veri_64/"
    verilist = os.listdir(veri_path)

    if continue_training == 1:
        print("load model from {}...".format(model_load_path))
        model = torch.load(model_load_path).to(device)
    else:
        model = Cnn_model()

    # difference = nn.MSELoss(reduce=False).to(device)
    # sym = Load_Symbol(symbol_save_path,out_channel)
    # dif = torch.zeros(number,n_classes)
    criterion = nn.TripletMarginLoss(margin=margin, p=2, eps=1e-8)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    amsoftmax = AMSoftmax(out_channel,2)

    viz.line([0.], [0], update='append', win='tr_loss', opts={'title': 'tr_loss'})
    loss = 0

    for epoch in range(number):
        print("epoch:", epoch + 1, "/", number)
        index = torch.randperm(len(verilist))
        for batch_id in range(len(verilist)):
            print("epoch", epoch + 1, " batch:", batch_id + 1, "/", len(verilist), end="   ")
            inp = torch.load(veri_path + verilist[index[batch_id]])

            inp = inp.to(device)
            inp1 = inp[:batch_size / 2]
            inp2 = inp[batch_size / 2:]
            output = model(inp1,batch_size,300,numc).to(device)
            # _,post = amsoftmax(output[:,0],output[:,1])
            # _,nega = -amsoftmax(output[:,0],output[:,2])
            # loss_batch = post.item() + nega.item()
            # optimizer.zero_grad()
            # post.backward()
            # nega.backward()
            # optimizer.step()
            loss_batch = criterion(output[:,0],output[:,1],output[:,2]).to(device)

            print("loss: ", loss_batch.item(),end="\n")
            loss += loss_batch.item()
            viz.line(torch.tensor([loss_batch]), [len(verilist) * epoch + batch_id], update='append',
                     win='tr_loss', opts={'title': 'tr_loss'})
            # logits = logits.expand(batch_size,sym.num,logits.size(-1))

            # optimizer.zero_grad()
            # loss_batch.backward()
            # optimizer.step()

        loss /= len(verilist)
        torch.save(model, model_save_path[0:-4]+"_"+str(loss)+".bin")


            # for j in range(min(batch_size,number-batch_size*batch_id)):
            #     #d = (abs(logits[j]-sym.user) / sym.user).sum(-1)/sym.user.size(-1)
            #     d = difference(logits[j], sym.user).sum(-1) /logits.size(-1)
            #     print("第",batch_size*batch_id+j+1,"项：")
            #     print("区别度向量：\n",d)
            #     print("区别最小：",torch.argmin(d),":",min(d))
            #     print(tgt[j].int().squeeze().tolist())
            #     print("区别最大：",torch.argmax(d), ":", max(d))
            #     print("区别度均值：",sum(d)/len(d))
            #     time.sleep(5)
            #     dif[batch_id*batch_size+j] = d


def matrix_similarity(mat1,mat2):
    pass

def veri_test(veri_test_file, model_fa_path, fa_data_dir, max_sec, step_sec, frame_step, dim,
         out_channel,n_classes,numc):
    print("Use {} for test".format(veri_test_file))

    audiolist, labellist = tools.get_veri_test_datalist(fa_data_dir + "test_veri/", veri_test_file)
    test_gene = tools.Loader_veri(audiolist, dim, max_sec, step_sec, frame_step,
                                   1, n_classes, labels=labellist, shuffle=True, mode = "test_veri")

    model_name_list = os.listdir(model_fa_path)
    min_eer = 100
    min_model = ""
    print("Start testing...")
    # cosine_distance = CosineSimilarity()
    for model_name in model_name_list:
        this_model_path = model_fa_path + model_name
        print("============================")
        print("Load model form {}".format(this_model_path))
        # Load model
        model = torch.load(this_model_path).to(device)
        model.eval()
        scores, labels = [], []
        for c in range(test_gene.batchs()):
            if c % 1000 == 0:
                print('Finish extracting features for {}/{}th wav.'.format(c, test_gene.batchs()))
            file_name, inp, tgt = test_gene.batch_data_mfcc_test(c)
            with torch.no_grad():
                # print(inp[0].shape)
                # print(inp[1].shape)
                output1 = model(inp[0].to(device),1,inp[0].size(-2),numc).to(device)
                output2 = model(inp[1].to(device),1,inp[1].size(-2),numc).to(device)
            output = torch.zeros([2, 1, out_channel])
            output[0] = output1
            output[1] = output2

            # dis = calculate_distance(output)
            dis = torch.cosine_similarity(output2,output1,dim=-1).mean()

            # if dis < 0.30 and tgt.cpu().detach().numpy() == 0:
            #     print("negative -> positive:",file_name[0][0],file_name[0][1])
            # elif dis > 0.30 and tgt.cpu().detach().numpy() == 1:
            #     print("positive -> negative:", file_name[0][0], file_name[0][1])

            scores.append(dis.cpu().detach().numpy())
            labels.append(tgt.cpu().detach().numpy())

        print(scores[0].shape)
        eer, thresh = tools.calculate_eer(labels, scores, model_name)

        print("EER: {}".format(eer))
        if eer < min_eer:
            min_eer = eer
            min_model = model_name


    # 输出最终结果
    print("============================")
    print("Min_eer : ", min_eer)
    print("Model_name : ", min_model)
    print("============================")

class Load_Symbol(object):
    def __init__(self,symbol_save_path,out_channel):
        print("loading symbol from",symbol_save_path)
        self.data = torch.load(symbol_save_path)
        self.num = len(self.data)
        self.user = torch.zeros(self.num,out_channel).to(device)
        for i in range(self.num):
            self.user[i] = self.data[str(i)]
            for j in range(self.user[i].size(-1)):
                if self.user[i][j].squeeze().tolist() == 0:
                    self.user[i][j] = torch.tensor(1e-2)

def calculate_distance(emb):
    distance = F.pairwise_distance(emb[0], emb[1], p=2) / 15
    return distance

if __name__ == "__main__":
    veri_vggvox_model(c.VERI_MODEL_PATH,c.FA_DIR,c.MARGIN,c.VERI_BATCH_SIZE,c.VERI_NUMBER,c.OUT_CHANNEL,
                      c.CONTINUE_TRAINING,c.VERI_LR,c.WEIGHT_DECAY,c.VERI_MODEL_LOAD_PATH,c.NUMC)
    veri_test(c.VERI_TEST_LIST_FILE,c.VERI_MODEL_FA_PATH,c.FA_DIR,c.MAX_SEC,
              c.BUCKET_STEP,c.FRAME_STEP,c.MFCC_DIM,c.OUT_CHANNEL,c.IDEN_CLASS,c.NUMC)

    # MIN EER: 0.17492303386509936