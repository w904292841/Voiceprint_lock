import torch
import constants as c
from torch import nn
from torch.nn import functional as F

class Conv_bn_pool(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,eps=1e-5,momentum=1,affine=True,
                 track_running_states=True,pool='',pool_size=None,pool_stride=None):
        super(Conv_bn_pool,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                              stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels,eps=eps,momentum=momentum,affine=affine,
                                  track_running_stats=track_running_states)
        self.act = nn.ReLU()
        self.pool = nn.Sequential()
        if pool=='max':
            self.pool = nn.MaxPool2d(pool_size,stride=pool_stride)
        elif pool=='avg':
            self.pool = nn.AvgPool2d(pool_size,stride=pool_stride)


    def forward(self, inp):
        output = self.conv(inp)
        output = self.bn(output)
        output = self.act(output)
        output = self.pool(output)
        return output


class Conv_bn_dynamic_apool(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,eps=1e-5,momentum=1,affine=True,

                 track_running_states=True):
        super(Conv_bn_dynamic_apool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                  track_running_stats=track_running_states)
        self.act = nn.ReLU()
        self.gapool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, inp):
        output = self.conv(inp)
        output = self.bn(output)
        output = self.act(output)
        output = self.gapool(output)
        # output = output.squeeze()
        return output
class Vggvox_model(nn.Module):

    def __init__(self,n_classes):
        super(Vggvox_model,self).__init__()
        inp = torch.zeros(c.INPUT_SHAPE)
        inp=inp.view(1,1,c.NUM_FFT,300)
        self.layer1 = Conv_bn_pool(in_channels=inp.size(1), out_channels=96, kernel_size=(7, 7), stride=(2, 2),
                                   padding=(1, 1), pool='max', pool_size=(3,3), pool_stride=(2,2))
        self.layer2 = Conv_bn_pool(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
                                   padding=(1, 1), pool='max', pool_size=(3,3), pool_stride=(2,2))
        self.layer3 = Conv_bn_pool(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))
        self.layer4 = Conv_bn_pool(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))
        self.layer5 = Conv_bn_pool(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1), pool='max', pool_size=(5, 3), pool_stride=(3, 2))
        self.layer6 = Conv_bn_dynamic_apool(in_channels=256, out_channels=512, kernel_size=(9, 1), stride=(1, 1),
                                   padding=(0, 0))
        self.layer7 = Conv_bn_pool(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1),
                                   padding=(0, 0))
        self.layer8 = Conv_bn_pool(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1),
                                   padding=(0, 0))
        # 归一化
        self.layer9 = Conv_bn_pool(in_channels=256, out_channels=n_classes, kernel_size=(1, 1), stride=(1, 1),
                                   padding=(0, 0))
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()
        # self.bn = nn.BatchNorm2d(256, eps=1e-5, momentum=1e-5, affine=True,
        #                           track_running_stats=True)
        # self.layer8= Conv_bn_pool(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1),
        #                                    padding=(0, 0))
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp, tgt, batch_size, n_classes):
        x = self.layer1(inp)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.layer7(x)
        # x = self.act(x)
        x = self.layer8(x)
        # x = self.act(x)
        # x = self.dropout(x)
        #归一化
        x = F.normalize(x,p=2,dim=1,eps=1e-12)
        # x = x.squeeze()
        x = self.layer9(x)
        logits = x.view(batch_size,n_classes)

        # loss = self.criterion(logits,tgt.long())
        predict = torch.argmax(F.softmax(logits,dim=1),dim=1,keepdim=False)
        correct = 0
        for i in range(batch_size):
            correct += 1 if predict[i] == tgt[i] else 0
        acc = float(correct)/batch_size
        return logits, acc


