import torch
import constants as c
from torch import nn
from torch.nn import functional as F

class Conv_bn_pool(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,eps=1e-5,momentum=1,affine=True,
                 track_running_states=True,pool='',pool_size=None,pool_stride=None,pool_padding=(0,0)):
        super(Conv_bn_pool,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                              stride=stride,padding=padding)
        nn.init.orthogonal_(self.conv.weight)
        self.pool = nn.Sequential()
        self.bn = nn.Sequential()
        self.act = nn.Sequential()
        if pool=='max':
            self.pool = nn.MaxPool2d(pool_size,stride=pool_stride,padding=pool_padding)
            self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_states)
            self.act = nn.ReLU()
        elif pool=='avg':
            self.pool = nn.AvgPool2d(pool_size,stride=pool_stride,padding=pool_padding)
            self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_states)
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
                              stride=stride, padding=padding)
        nn.init.orthogonal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                  track_running_stats=track_running_states)
        self.act = nn.ReLU()
        self.pooling = nn.MaxPool2d((2,2),stride=(2,2),padding=pool_padding)
        self.gapool = nn.AdaptiveAvgPool2d((1,1))

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
class Vggvox_model(nn.Module):

    def __init__(self,device = torch.device('cuda')):
        super(Vggvox_model,self).__init__()
        inp = torch.zeros(c.INPUT_SHAPE)
        inp=inp.view(1,1,c.NUM_FFT,300)
        self.layer1 = Conv_bn_pool(in_channels=inp.size(1), out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(0, 1), pool='max', pool_size=(2, 2), pool_stride=(2, 2),
                                   pool_padding=(1, 1))
        self.layer2 = Conv_bn_pool(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(0, 1), pool='max', pool_size=(2, 2), pool_stride=(2, 2),
                                   pool_padding=(1, 1))
        self.layer3_1 = Conv_bn_pool(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1))
        self.layer3_2 = Conv_bn_pool(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 0), pool='max', pool_size=(2, 2), pool_stride=(2, 2),
                                     pool_padding=(1, 1))
        self.layer4 = Conv_bn_dynamic_apool(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 2),
                                            padding=(1, 1))
        # self.layer5 = Conv_bn_pool(in_channels=384, out_channels=384, kernel_size=(1, 1), stride=(1, 1),
        #                            padding=(0, 0))
        # #归一化
        #
        # self.layer6 = Conv_bn_pool(in_channels=384, out_channels=256, kernel_size=(1, 1), stride=(1, 1),
        #                                    padding=(0, 0))
        #self.layer8 = Conv_bn_pool(in_channels=1024, out_channels=n_classes, kernel_size=(1, 1), stride=(1, 1),
        #                           padding=(0, 0))
        for p in self.parameters():
            p.requires_grad = False
        self.layer9 = nn.Linear(in_features=256, out_features=256)
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp, batch_size, out_channel):

        num = inp.size(1)
        x = self.layer1(inp.view(batch_size * num, 1, c.NUM_FFT, 300))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.layer6(x)
        # x = self.layer7(x)
        #归一化
        # x = self.layer8(x)
        x = F.normalize(x, p=2, dim=2, eps=1e-12)
        x = x.squeeze()
        x = x.view(batch_size, num, -1)

        # right_cos = torch.zeros([batch_size])
        # wrong_cos = torch.zeros([batch_size])
        # for i in range(batch_size):
        #     right_cos[i] = torch.dot(x[i, 0], x[i, 1])
        #     wrong_cos[i] = torch.dot(x[i, 0], x[i, 2])
        #
        # loss = F.relu(margin + wrong_cos - right_cos).clone().detach().requires_grad_(True)

        return x


