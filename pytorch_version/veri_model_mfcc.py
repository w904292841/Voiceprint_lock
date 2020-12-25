import torch
import constants as c
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda")

class Conv_bn_pool(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,eps=1e-5,momentum=1,affine=True,
                 track_running_states=True,pool='',pool_size=None,pool_stride=None,pool_padding=(0,0)):
        super(Conv_bn_pool,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                              stride=stride,padding=padding).to(device)
        nn.init.orthogonal_(self.conv.weight)
        self.pool = nn.Sequential().to(device)
        self.bn = nn.Sequential().to(device)
        self.act = nn.Sequential().to(device)
        if pool=='max':
            self.pool = nn.MaxPool2d(pool_size,stride=pool_stride,padding=pool_padding).to(device)
            self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_states).to(device)
            self.act = nn.ReLU()
        elif pool=='avg':
            self.pool = nn.AvgPool2d(pool_size,stride=pool_stride,padding=pool_padding).to(device)
            self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_states).to(device)
            self.act = nn.ReLU()


    def forward(self, inp):
        output = self.conv(inp)
        output = self.bn(output)
        output = self.act(output)
        output = self.pool(output)

        # print(output.shape)
        return output


class Conv_bn_dynamic_apool(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,eps=1e-5,momentum=1,affine=True,
                 track_running_states=True,pool_padding=(1,1)):
        super(Conv_bn_dynamic_apool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding).to(device)
        nn.init.orthogonal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                  track_running_stats=track_running_states).to(device)
        self.act = nn.ReLU()
        self.pooling = nn.MaxPool2d((2,2),stride=(2,2),padding=pool_padding).to(device)
        self.gapool = nn.AdaptiveAvgPool2d((1,1)).to(device)

    def forward(self, inp):
        output = self.conv(inp)
        output = self.bn(output)
        output = self.act(output)
        # print(output.shape)
        output = self.pooling(output)
        output = self.gapool(output)
        output = output.squeeze()
        # print(output.shape)
        return output

class Cnn_model(nn.Module):

    def __init__(self):
        super(Cnn_model,self).__init__()
        inp = torch.zeros(c.MFCC_DIM)
        inp=inp.unsqueeze(0)
        self.layer1 = Conv_bn_pool(in_channels=inp.size(1), out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(0, 1), pool='max', pool_size=(2, 2), pool_stride=(2, 2),
                                   pool_padding=(1, 1))
        self.layer2 = Conv_bn_pool(in_channels=96, out_channels=192, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(0, 1), pool='max', pool_size=(2, 2), pool_stride=(2, 2),
                                   pool_padding=(1, 1))
        self.layer3_1 = Conv_bn_pool(in_channels=192, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1))
        self.layer3_2 = Conv_bn_pool(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 0), pool='max', pool_size=(2, 2), pool_stride=(2, 2),
                                     pool_padding=(1, 1))
        self.layer4 = Conv_bn_dynamic_apool(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 2),
                                            padding=(1, 1))


        self.layer5_veri = nn.Linear(512,512,bias=True).to(device)
        nn.init.orthogonal_(self.layer5_veri.weight)
        # self.layer6 = nn.Linear(256,n_classes,bias=True)
        # nn.init.orthogonal_(self.layer6.weight)
        # self.layer7 = Conv_bn_dynamic_apool(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1),
        #                            padding=(0, 0))
        # self.layer8 = Conv_bn_pool(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1),
        #                            padding=(0, 0))
        # #归一化
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()
        # # self.layer8= Conv_bn_pool(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1),
        # #                                    padding=(0, 0))
        # self.layer9 = Conv_bn_pool(in_channels=512, out_channels=n_classes, kernel_size=(1, 1), stride=(1, 1),
        #                            padding=(0, 0))
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp, batch_size, seconds, numc = 13):
        num = inp.size(1)
        inp = inp.view(batch_size * num, 3, seconds, numc)
        x = self.layer1(inp)
        x = self.layer2(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        # # 归一化
        x = F.normalize(x,p=2,dim=1,eps=1e-12)
        x = self.layer4(x)
        # print(x.shape)
        # x = self.layer5_veri(x)
        # x = self.act(x)
        # # x = self.dropout(x)
        #
        # x = self.layer6(x)
        # print(x.shape)
        logits = x.view(batch_size,num,-1)

        # loss = self.criterion(logits,tgt.long())
        # predict = torch.argmax(F.softmax(logits,dim=1),dim=1,keepdim=False)
        # correct = 0
        # for i in range(batch_size):
        #     correct += 1 if predict[i] == tgt[i] else 0
        # acc = float(correct)/batch_size
        return logits



if __name__ == "__main__":
    model = Cnn_model()
    data = torch.rand([128,3,3,300,16]).to(device)
    data = torch.load(r"H:\科研\mfcc_data_pre\train_veri\id10001_0_1.pt").unsqueeze(0).unsqueeze(0).to(device)
    print(data.shape)
    label = torch.rand([128])
    logits = model(data,1,300,16)

