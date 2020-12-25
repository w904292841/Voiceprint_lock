import os
import time
import torch
import visdom  #(可视化工具包)下载：pip install visdom -i https://pypi.tuna.tsinghua.edu.cn/simple
from model import Vggvox_model
import constants as c
import tools
from torch import optim
from torch import nn
import numpy as np
from wav_reader import get_fft_spectrum
from tools import build_buckets

viz=visdom.Visdom(env = 'vgg vox_iden_128')  #运行程序前命令行输入： python -m visdom.server

def main(trainfile, model_load_path, fa_data_dir, model_save_path, max_sec, step_sec, frame_step, dim, lr,
         continue_training, save_model, testfile, train_batch_size, epochs,test_batch_size,
         n_classes,weight_save_path):
    audiolist, labellist = tools.get_voxceleb1_datalist(fa_data_dir, trainfile)

    device = torch.device('cuda')

    if continue_training == 1:
        print("load model from {}...".format(model_load_path))
        model = torch.load(model_load_path).to(device)
    else:
        model = Vggvox_model(n_classes).to(device)
        # 编译模型
    optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    acc_highest = 0
    print("Start training...")
    viz.line([0.], [0], update='append', win='tr_loss', opts={'title': 'tr_loss'})
    viz.line([0.], [0], update='append', win='tr_acc', opts={'title': 'tr_acc'})
    viz.line([0.], [0], update='append', win='test_loss', opts={'title': 'test_loss'})
    viz.line([0.], [0], update='append', win='test_acc', opts={'title': 'test_acc'})
    for epoch in range(epochs):
        train_gene = tools.Loader(audiolist, labellist, dim, max_sec, step_sec, frame_step,
                                  train_batch_size, n_classes)
        loss = 0
        acc=0
        acc_high=0
        print("epoch:",epoch + 1,"/",epochs)
        for batch_id in range(train_gene.batchs()):
            print("epoch",epoch + 1," batch:",batch_id + 1,"/",train_gene.batchs())
            inp, tgt = train_gene.batch_data(batch_id)
            inp, tgt = inp.to(device), tgt.to(device)
            logits,acc_batch = model(inp, tgt, train_batch_size, n_classes)
            loss_batch = criterion(logits,tgt.long()).to(device)
            print("loss: ", loss_batch.item())
            print("acc: ", acc_batch)
            viz.line(torch.tensor([loss_batch]), [train_gene.batchs()*epoch+batch_id], update='append', win='tr_loss', opts={'title': 'tr_loss'})
            viz.line(torch.tensor([acc_batch]), [train_gene.batchs()*epoch+batch_id], update='append', win='tr_acc', opts={'title': 'tr_acc'})
            acc_high = acc_batch if acc_batch > acc_high else acc_high

            loss += loss_batch.item()
            acc += acc_batch

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        loss /= train_gene.batchs()
        acc /= train_gene.batchs()
        if acc >= acc_highest:
            acc_highest = acc
            if save_model:
                torch.save(model, model_save_path)
                state_dic = {
                    "state": model.state_dict(),
                    "epoch": epoch
                }
                if not os.path.isdir(weight_save_path+'/checkpoint'):
                    os.mkdir(weight_save_path+'/checkpoint')
                torch.save(state_dic, weight_save_path+'/checkpoint/autoencoder.t7')

        print("epoch",epoch + 1,":")
        print("loss:",loss)
        print("acc:",acc)
        print("acc_high:",acc_high)

        iden(testfile,fa_data_dir,model_save_path,max_sec,step_sec,frame_step,dim, test_batch_size,n_classes, epoch)


    print("Training completed！")
    print("highest acc:",acc_highest)


def iden(testfile,fa_data_dir,iden_model,max_sec, step_sec, frame_step, dim, batch_size, n_classes, epoch):
    # 读入测试数据、标签
    print("Use {} for test".format(testfile))

    iden_list = np.loadtxt(testfile, str,delimiter=",")

    voice_list = np.array([os.path.join(fa_data_dir, i[0]) for i in iden_list])
    total_length = len(voice_list)
    device = torch.device('cuda')
    labels = torch.tensor([int(i[1]) for i in iden_list]).to(device)
    criterion = nn.CrossEntropyLoss()

    # Load model
    print("Load model form {}".format(iden_model))
    model = torch.load(iden_model).to(device)

    print("Start identifying...")

    acc = 0
    loss = 0
    model.eval()
    for num, ID in enumerate(voice_list):
        if num % 100 == 0:
            print('Finish identifying for {}/{}th wav.'.format(num, total_length))
        b_data = torch.tensor(
            get_fft_spectrum(ID, build_buckets(max_sec, step_sec, frame_step),mode="test").tolist()).to(device)
        b_data = b_data.unsqueeze(0).to(device)
        with torch.no_grad():
            eval_predict, tmp_eval_accuracy = model(b_data.unsqueeze(0), labels[num].unsqueeze(0), 1, n_classes)
        tmp_eval_loss = criterion(eval_predict, labels[num].unsqueeze(0).long()).to(device)
        loss += tmp_eval_loss.item()
        acc += tmp_eval_accuracy

    loss /= total_length
    acc /= total_length
    viz.line(torch.tensor([loss]), [epoch], update='append', win='test_loss',
             opts={'title': 'test_loss'})
    viz.line(torch.tensor([acc]), [epoch + 1], update='append', win='test_acc',
             opts={'title': 'test_acc'})

    print("eval_loss:",loss)
    print("eval_acc:",acc)

if __name__ =='__main__':
    '''
    训练
    每个epoch测试一次
    '''
    print(
        "*****Check params*****\nlearn_rate:{}\nepochs:{}\nbatch_size:{}\nclass_num:{}\ncontinue_training:{}\nsave_model:{}\n*****Check params*****"
        .format(c.LR, c.EPOCHS, c.BATCH_SIZE, c.IDEN_CLASS, c.CONTINUE_TRAINING, c.SAVE))
    time.sleep(15)
    # set_learning_phase(0)
    main(c.IDEN_TRAIN_LIST_FILE, c.IDEN_MODEL_PATH, c.FA_DIR, c.IDEN_MODEL_PATH, c.MAX_SEC, c.BUCKET_STEP,
                       c.FRAME_STEP, c.DIM, c.LR, c.CONTINUE_TRAINING, c.SAVE,c.IDEN_TEST_FILE, c.BATCH_SIZE,
                       c.EPOCHS, c.TEST_BATCH_SIZE, c.IDEN_CLASS,c.IDEN_MODEL_FA_PATH)
    # iden(c.IDEN_TEST_FILE,c.FA_DIR,c.IDEN_MODEL_PATH,c.MAX_SEC,c.BUCKET_STEP,c.FRAME_STEP,c.MFCC_DIM,
    #      c.TEST_BATCH_SIZE,c.IDEN_CLASS,c.EPOCHS)
