from turtle import forward
import torch
from torch import nn
from modules import ConvSC, Inception
from swin_transformer import SwinTransformer3D

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
        m = nn.Upsample(scale_factor=8, mode='nearest')
        hid = m(hid)
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
        self.hid = Mid_Xnet(int(T*hid_S/2), hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, 3, N_S)
        # xly add for backbone
        self.skip_layer = ConvSC(3, int(hid_S/2), stride=1)
        # xly add for input of different chann
        self.conv_input = nn.Conv2d(shape_in[1], 3, kernel_size=3, stride=1, padding=1)


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
        embed = self.backbone(x_raw)
        embed = embed.transpose(1,2)

        #print("shape of embed ", embed.shape)
        # embed, skip = self.enc(x)
        _, _, C_, H_, W_ = embed.shape
        T_ = T/2
        z = embed.view(B, int(T_), C_, H_, W_)
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


