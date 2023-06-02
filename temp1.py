import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from models.model_new import Encoder, conv_lstm, gaussian_conv_lstm, swish, DecoderBlock
from models.model_LMC_modify import conv_lstm, Encoder, DecoderBlock, gaussian_conv_lstm
from torchsummary import summary
import random
from args import get_parser
# import flopth
import time
from flopth import flopth

class Decoder(nn.Module):
    def __init__(self, g_dim=512, init_filters=64, skip_type='residual', n_channel=3):
        super(Decoder, self).__init__()
        self.skip_type = skip_type
        self.n_channel = n_channel
        self.g_dim = g_dim
        if self.skip_type == 'residual':
            self.layer1 = DecoderBlock(in_filters=init_filters, out_filters=init_filters, upsample=True)
        else:
            self.layer1 = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters, upsample=True)
        self.layer2 = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters)
        if self.skip_type == 'residual':
            self.layer3 = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2, upsample=True)
        else:
            self.layer3 = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters * 2, upsample=True)
        self.layer4 = DecoderBlock(in_filters=init_filters * 3, out_filters=init_filters * 2)
        if self.skip_type == 'residual':
            self.layer5 = DecoderBlock(in_filters=init_filters * 3, out_filters=init_filters * 3, upsample=True)
        else:
            self.layer5 = DecoderBlock(in_filters=init_filters * 6, out_filters=init_filters * 3, upsample=True)
        self.layer6 = DecoderBlock(in_filters=self.g_dim, out_filters=init_filters * 3)
        self.conv_out = nn.Conv2d(init_filters, self.n_channel, kernel_size=7, stride=1, padding=3)

    def forward(self, x, skips1, skips2, skips3):
        y = x
        y = self.layer6(y)                          # 192x8x8
        if self.skip_type == 'residual':
            y = y + skips1
        else:
            y = torch.cat([y, skips1], dim=1)
        y = self.layer5(y)                          # 192x16x16
        y = self.layer4(y)                          # 128x16x16
        if self.skip_type == 'residual':
            y = y + skips2
        else:
            y = torch.cat([y, skips2], dim=1)
        y = self.layer3(y)                          # 128x32x32
        y = self.layer2(y)                          # 64x32x32
        if self.skip_type == 'residual':
            y = y + skips3
        else:
            y = torch.cat([y, skips3], dim=1)
        y = self.layer1(y)                          # 64x64x64
        y = self.conv_out(y)                        # n_channelx64x64
        y = torch.sigmoid(y)
        return y


# sch_sampling = 10
# epoch = 34
# sc_prob = sch_sampling / (sch_sampling + np.exp(epoch / sch_sampling))
# print(sc_prob)


# encoder = Encoder(g_dim=192, n_channel=6, activation_type='relu')
# summary(encoder, input_size=(6, 64, 64))
# x = torch.zeros(size=(1, 6, 64, 64))
# start_time = time.time()
# out = encoder(x)
# end_time = time.time()
# print('Running time: ', end_time-start_time)


# posterior = gaussian_conv_lstm(input_size=192, output_size=50, hidden_size = 128, n_layers=1, batch_size=2, image_size = (8,8), device='cpu')
# posterior.init_hidden()
# x = torch.zeros(size=(2, 192, 8, 8))
# start_time = time.time()
# out = posterior(x)
# end_time = time.time()
# print('Running time: ', end_time-start_time)
# posterior.hidden = posterior.init_hidden(device='cpu')
# summary(posterior, input_size=(192,8,8), batch_size=2)
# flops, params = flopth(posterior, in_size=((192, 8, 8),),inputs=inp_x, show_detail=True)

# frame_predictor = conv_lstm(192+50, output_size= 192, hidden_size = 128, n_layers=2, batch_size=2, image_size=(8, 8), device='cpu')
# x = torch.zeros(size=(2, 192+50, 8, 8))
# start_time = time.time()
# out = frame_predictor(x)
# end_time = time.time()
# print('Running time: ', end_time-start_time)
# frame_predictor.init_hidden()
# summary(frame_predictor, input_size=(192+50, 8, 8), batch_size=2)

# decoder = Decoder(192, skip_type='residual', n_channel=6)
# x = torch.zeros(size=(1, 192, 8, 8))
# h1 = torch.zeros(size=(1, 192, 8, 8))
# h2 = torch.zeros(size=(1, 128,16,16))
# h3 = torch.zeros(size=(1, 64, 32, 32))
# start_time = time.time()
# out = decoder(x, h1, h2, h3)
# end_time = time.time()
# print('Running time: ', end_time-start_time)
# summary(decoder, input_size=[(192, 8, 8), (192, 8, 8), (128,16,16), (64, 32, 32)])
# flops, params = flopth(decoder, in_size=((192, 8, 8), (192, 8, 8), (128,16,16), (64, 32, 32)), show_detail=True)
# print(flops)
# print(params)

#-----------------------------------------SLAMP------------------------------------------------
from models.model_slamp import encoder, decoder
# Enc = encoder(dim=128)
# start_time = time.time()
# out, [h1, h2, h3, h4, h5] = Enc(torch.zeros(size=(1, 3, 92, 310)))
# end_time = time.time()
# print('Running time: ', end_time-start_time)
# print(h1.shape)
# print(h2.shape)
# print(h3.shape)
# print(h4.shape)
# print(h5.shape)
#
Dec = decoder(dim=128)
# x = torch.zeros(size=(1, 128, 4, 4))
# start_time = time.time()
# out = Dec(x, h1, h2, h3, h4, h5)
# end_time = time.time()
# print('Running time: ', end_time-start_time)
# summary(Dec, input_size=[(128, 4, 4), (64, 46, 156), (96, 24, 80), (128, 12, 40), (192, 6, 20), (256, 4, 4)])
flops, params = flopth(Dec, in_size=((128, 4, 4), (64, 46, 156), (96, 24, 80), (128, 12, 40), (192, 6, 20), (256, 4, 4)), show_detail=True)
print(flops)
print(params)

#------------------------------SimVP----------------------------------------------
# from models.model_simvp import SimVP
# model = SimVP(tuple([10, 1, 64, 64]), 64, 256, 4, 8)
# x = torch.zeros(size=(1, 10, 1, 64, 64))
# start_time = time.time()
# out = model(x)
# end_time = time.time()
# print('Running time: ', end_time-start_time)
# summary(model, input_size=(10, 1, 64, 64))
# flops, params = flopth(model, in_size=((10, 1, 64, 64), ), show_detail=True)
# print(flops)
# print(params)

#-----------------------------SVG---------------------------------------------
from models.model_svg import encoder, decoder, lstm
# Enc = encoder(dim=128)
# out, [h1, h2, h3, h4, h5] = Enc(torch.zeros(size=(2, 1, 128, 128)))
# print(h1.shape)
# print(h2.shape)
# print(h3.shape)
# print(h4.shape)
# print(h5.shape)
# summary(Enc, input_size=(1, 128, 128))
# flops, params = flopth(Enc, in_size=((1, 128, 128)), show_detail=True)
# print(flops)
# print(params)
# Dec = decoder(dim=128)
# summary(Dec, input_size=[(128, ), (64, 64, 64), (128, 32, 32), (256, 16, 16), (512, 8, 8), (512, 4, 4)])
# flops, params = flopth(Dec, in_size=((128, ), (64, 64, 64), (128, 32, 32), (256, 16, 16), (512, 8, 8), (512, 4, 4)), show_detail=True)
# print(flops)
# print(params)
# predictor = lstm(128+50, 128, 256, 2, batch_size=2)
# summary(predictor, input_size=(4, 128+10, ), batch_size=4)
# flops, params = flopth(predictor, in_size=((2, 128+50,),), show_detail=True)
# print(flops)
# print(params)
