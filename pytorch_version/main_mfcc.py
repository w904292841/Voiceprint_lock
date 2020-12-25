import os
import datetime
# from model_my import resnet34
import torch
import visdom  #(可视化工具包)下载：pip install visdom -i https://pypi.tuna.tsinghua.edu.cn/simple
from mfcc_model import Cnn_model
import constants as c
import tools
from torch import optim
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

viz=visdom.Visdom(env = 'cnn+mfcc vox_iden')  #运行程序前命令行输入： python -m visdom.server

def main(model_load_path, fa_data_dir, model_save_path, max_sec, step_sec, frame_step, dim, lr,
         continue_training, save_model, testfile, train_batch_size, epochs,test_batch_size,n_classes,
         weight_save_path,weight_decay):
    datalist = os.listdir(fa_data_dir + "train_batch")

    device = torch.device('cuda')

    if continue_training == 1:
        print("load model from {}...".format(model_load_path))
        model = torch.load(model_load_path).to(device)
    else:
        model = Cnn_model(n_classes).to(device)
        # 编译模型


    optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=weight_decay)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    acc_highest = 0
    print("Start training...")
    viz.line([0.], [0], update='append', win='tr_loss', opts={'title': 'tr_loss'})
    viz.line([0.], [0], update='append', win='tr_acc', opts={'title': 'tr_acc'})
    viz.line([0.], [0], update='append', win='dev_loss', opts={'title': 'dev_loss'})
    viz.line([0.], [0], update='append', win='dev_acc', opts={'title': 'dev_acc'})


    test_highest_acc = 0

    train_loss_p = []
    train_acc_p = []
    test_loss_p = []
    test_acc_p = []

    for epoch in range(epochs):
        loss = 0
        acc=0
        acc_high=0
        index = torch.randperm(len(datalist))
        print("epoch:",epoch + 1,"/",epochs)
        model.train()
        for batch_id in range(len(datalist)):
            print("epoch",epoch + 1," batch:",batch_id + 1,"/",len(datalist),end="    ")
            inp, tgt = torch.load(fa_data_dir  + "train_batch/" + datalist[index[batch_id]])
            inp, tgt = inp.to(device), tgt.to(device)

            logits, acc_batch, _, _ = model(inp,tgt,train_batch_size,n_classes)

            loss_batch = criterion(logits, tgt.long()).to(device)
            print("loss: ", loss_batch.item(),end="   ")
            print("acc: ", acc_batch)

            train_loss_p.append(loss_batch.item())
            train_acc_p.append(acc_batch)
            viz.line(torch.tensor([loss_batch]), [len(datalist)*epoch+batch_id], update='append', win='tr_loss', opts={'title': 'tr_loss'})
            viz.line(torch.tensor([acc_batch]), [len(datalist)*epoch+batch_id], update='append', win='tr_acc', opts={'title': 'tr_acc'})
            acc_high = acc_batch if acc_batch > acc_high else acc_high

            loss += loss_batch.item()
            acc += acc_batch

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        # scheduler.step(epoch)

        loss /= len(datalist)
        acc /= len(datalist)
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

        print("epoch",epoch + 1,":     loss:",loss,"   acc:",acc,"  acc_high:",acc_high)

        test_highest_acc, test_acc, test_loss = iden(testfile,fa_data_dir,model_save_path,max_sec,step_sec,frame_step,dim,
                                test_batch_size,lr,n_classes,epoch, test_highest_acc,model_save_path[0:-4])
        test_loss_p.append(test_loss)
        test_acc_p.append(test_acc)

    # plt.plot(train_loss_p, 'b', label="Training Loss")  # bo表示蓝色原点
    # # plt.plot(ep, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    # plt.title("Train Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig("../img/train_loss" + datetime.datetime.now().strftime('%Y-%m-%d') + ".img")
    #
    # plt.plot(train_acc_p, 'b', label="Training Acc")  # bo表示蓝色原点
    # # plt.plot(ep, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    # plt.title("Train Acc")
    # plt.xlabel("Epochs")
    # plt.ylabel('Acc')
    # plt.legend()
    # plt.savefig("../img/train_acc" + datetime.datetime.now().strftime('%Y-%m-%d') + ".img")
    #
    # plt.plot(test_loss_p, 'b', label="Test Loss")  # bo表示蓝色原点
    # # plt.plot(ep, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    # plt.title("Test Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig("../img/test_loss" + datetime.datetime.now().strftime('%Y-%m-%d') + ".img")
    #
    # plt.plot(test_acc_p, 'b', label="Test Acc")  # bo表示蓝色原点
    # # plt.plot(ep, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    # plt.title("Test Acc")
    # plt.xlabel("Epochs")
    # plt.ylabel('Acc')
    # plt.legend()
    # plt.savefig("../img/test_acc" + datetime.datetime.now().strftime('%Y-%m-%d') + ".img")


    print("Training completed！")
    print("highest acc:",acc_highest)


def iden(testfile,fa_data_dir,iden_model,max_sec, step_sec, frame_step, dim, batch_size, lr,
         n_classes,epoch, highest_acc,iden_fa_model):
    # 读入测试数据、标签
    print("Use {} for dev".format(testfile))

    audiolist, labellist, numlist = tools.get_voxceleb1_datalist(fa_data_dir + "dev/", testfile)

    device = torch.device('cuda')
    test_gene = tools.Loader_pt(audiolist, labellist, dim, max_sec, step_sec, frame_step, 1, n_classes, mode = "dev")
    criterion = nn.CrossEntropyLoss()

    # Load model
    print("Load model form {}".format(iden_model))
    model = torch.load(iden_model).to(device)

    print("Start identifying...")

    recognize_wrong = []
    recognize_wrong_to = []

    for i in range(n_classes):
        recognize_wrong.append(0)
        recognize_wrong_to.append(0)

    # test_data = tools.Loader(voice_list, labels, dim, max_sec, step_sec, frame_step, batch_size,
    #                           n_classes,shuffle=False,mode="test")

    acc = 0
    loss = 0
    model.eval()
    for num in range(test_gene.batchs()):
        if num % 100 == 0: print('Finish identifying for {}/{}th wav.'.format(num, test_gene.batchs()))
        inp, tgt = test_gene.batch_data_mfcc(num)
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.no_grad():
            logits, correct, wrong, wrong_to = model(inp,tgt,1,n_classes)

        if wrong <n_classes:
            recognize_wrong[wrong] += 1
            recognize_wrong_to[wrong_to] += 1

        acc += correct
        tmp_loss = criterion(logits,tgt.long()).to(device)
        loss += tmp_loss.item()

    # for batch_id in range(test_data.batchs()):
    #     inp, tgt = test_data.batch_data_mfcc(batch_id)
    #     inp, tgt = inp.to(device), tgt.to(device)
    #
    #     with torch.no_grad():
    #         logits, tmp_eval_accuracy= model(inp,tgt,batch_size,n_classes)
    #
    #
    #     tmp_eval_loss = criterion(logits,tgt.long()).to(device)
    #     loss += tmp_eval_loss.item()
    #     acc += tmp_eval_accuracy

    loss /= test_gene.batchs()
    acc /= test_gene.batchs()
    viz.line(torch.tensor([loss]), [epoch], update='append', win='dev_loss',
            opts={'title': 'dev_loss'})
    viz.line(torch.tensor([acc]), [epoch + 1], update='append', win='dev_acc',
            opts={'title': 'dev_acc'})

    print("eval_loss:",loss)
    print("eval_acc:",acc)

    # recognize_wrong = np.array(recognize_wrong)
    # recognize_wrong_to = np.array(recognize_wrong_to)
    #
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    #
    # label_list = range(n_classes)  # 横坐标刻度显示值
    # num_list1 = recognize_wrong / numlist  # 纵坐标值1
    # num_list2 = recognize_wrong_to / test_gene.batchs() * 20 # 纵坐标值2
    # x = range(len(num_list1))
    # """
    # 绘制条形图
    # left:长条形中点横坐标
    # height:长条形高度
    # width:长条形宽度，默认值0.8
    # label:为后面设置legend准备
    # """
    # rects1 = plt.bar(left=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="识别错误率")
    # rects2 = plt.bar(left=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="错误识别率*20")
    # plt.ylim(0, 1)  # y轴取值范围
    # plt.ylabel("单类别错误率")
    # """
    # 设置x轴刻度显示值
    # 参数一：中点坐标
    # 参数二：显示值
    # """
    # plt.xticks([index + 0.2 for index in x], label_list)
    # plt.xlabel("类别")
    # plt.title("识别错误率和错误识别率")
    # plt.legend()  # 设置题注
    # # 编辑文本
    # for rect in rects1:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    # for rect in rects2:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    # plt.show()
    #
    # image_name = iden_model[:-23] + "recognize_wrong.jpg"
    #
    # print("=> saving {}".format(image_name))
    # plt.savefig(image_name)

    if acc >= highest_acc:
        highest_acc = acc
        model_save_path = iden_fa_model + "_lr" + str(lr) + "_acc" + str(round(acc,4)) + "_loss" + str(round(loss,4)) + ".bin"
        torch.save(model,model_save_path)

    return highest_acc, acc, loss

if __name__ =='__main__':
    '''
    训练
    每个epoch测试一次
    '''
    print(
        "*****Check params*****\nlearn_rate:{}\nepochs:{}\nbatch_size:{}\nclass_num:{}\ncontinue_training:{}\nsave_model:{}\n*****Check params*****"
        .format(c.LR, c.EPOCHS, c.BATCH_SIZE, c.IDEN_CLASS, c.CONTINUE_TRAINING, c.SAVE))
    # # set_learning_phase(0)
    main(c.MFCC_IDEN_MODEL_PATH, c.FA_DIR, c.MFCC_IDEN_MODEL_PATH, c.MAX_SEC, c.BUCKET_STEP,c.FRAME_STEP,
         c.MFCC_DIM, c.LR, c.CONTINUE_TRAINING, c.SAVE,c.IDEN_TEST_FILE, c.BATCH_SIZE,
         c.EPOCHS, c.TEST_BATCH_SIZE, c.IDEN_CLASS,c.IDEN_MODEL_FA_PATH,c.WEIGHT_DECAY)
    # iden(c.IDEN_TEST_FILE,c.FA_DIR,c.MFCC_IDEN_MODEL_PATH,c.MAX_SEC,c.BUCKET_STEP,
    #    c.FRAME_STEP,c.MFCC_DIM,c.TEST_BATCH_SIZE,1e-3,c.IDEN_CLASS,1,1,c.IDEN_MODEL_FA_PATH)
