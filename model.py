from turtle import forward
import torch
from torch import nn
from modules import ConvSC, Inception
from swin_transformer import SwinTransformer3D
import math

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1

class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        # m = nn.Upsample(scale_factor=4, mode='nearest')
        # hid = m(hid)
        # hid = hid.view(*enc1.shape)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y

class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return nn.functional.upsample(x, size=(H,W), mode='bilinear') + y
    
    def forward(self,skips):
        new_skips = []
        for i in skips:
            b,c,t,h,w = i.shape
            i = i.transpose(1,2)
            i = i.reshape(b*t,c,h,w)
            new_skips.append(i)

        C1, C2, C3, C4, C5 = new_skips
        P5 = self.toplayer(C5)
        P4 = self._upsample_add(P5, self.latlayer1(C4))
        P3 = self._upsample_add(P4, self.latlayer2(C3))
        P2 = self._upsample_add(P3, self.latlayer3(C2))
        P1 = self._upsample_add(P2, self.latlayer4(C1))
        # Smooth
        P4 = self.smooth1(P4)
        P3 = self.smooth2(P3)
        P2 = self.smooth3(P2)
        P1 = self.smooth4(P1)

        return P1





       

class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        # 构建SimVP数据集
        # inshape [10, 1, 64, 64]
        # hid_S   64
        # hid_T   256
        # N_S     4
        # N_T     8
        T, C, H, W = shape_in
        self.backbone = SwinTransformer3D(pretrained='/workspace/weight/bevt_swin_base.pth')
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(int(T*256/2), hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(256, 3, N_S)
        # xly add for backbone
        self.skip_layer = ConvSC(3, 128, stride=1)
        # xly add for input of different chann
        self.conv_input = nn.Conv2d(shape_in[1], 3, kernel_size=3, stride=1, padding=1)
        self.fpn = Pyramid()


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x_raw = x_raw.view(B*T, C, H, W)
        x_raw = self.conv_input(x_raw)
        x_raw = x_raw.view(B,T, 3, H, W)
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)
        skip = self.skip_layer(x)   # shape of the skip is 20,64,64,64
        skip = skip.view(B*int(T/2),-1,H,W)
        # skip = skip[:,:int(T/2)] + skip[:,int(T/2):] # 将skip的维度与后面hid对齐
        x_raw = x_raw.transpose(1, 2)
        embed,skips = self.backbone(x_raw)
        
        # 在此处将skips和embed融合成为更好的金字塔特征
        feature = self.fpn(skips)
        embed = embed.transpose(1,2)

        #print("shape of embed ", embed.shape)
        # embed, skip = self.enc(x)
        # _, _, C_, H_, W_ = embed.shape
        # T_ = T/2
        # z = embed.view(B, int(T_), C_, H_, W_)
        _, C_, H_, W_ = feature.shape
        z = feature.reshape(B,int(T/2),C_,H_,W_)
        hid = self.hid(z)
        hid = hid.reshape(int(B*T/2), C_, H_, W_)
        # 此处的hid dim0=BxT/2,上面的skip的维度需要做出对应改变
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, int(T/2), C, H, W)
        #print("shape of out BTCHW ", Y.shape)

        return Y

if __name__ == "__main__":
    # 构建SimVP数据集
    #inshape = [8, 768, 7, 7]
    inshape = [10,3,64,64]
    hid_S = 1024
    hid_T = 256
    N_S = 4
    N_T = 8

    model = SimVP(inshape,hid_S,hid_T)
    model.eval()
    input = torch.randn([16,10,3,64,64])
    out = model(input)
    print(out.shape)


