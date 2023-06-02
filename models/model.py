import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Resize
from torch.autograd import Variable


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
    def forward(self, x):
        return x * nn.Sigmoid()(x)

class SEBlock(nn.Module):
    def __init__(self, c, r=6):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _,  _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters, strides=1):
        super(EncoderBlock, self).__init__()
        self.strides = strides
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=strides, padding=1)
        self.act = swish()
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.se_block = SEBlock(c=out_filters)
        self.residual = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides)
        self.bn_residual = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        residual = x
        y = self.bn1(x)
        y = self.act(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.se_block(y)
        if x.shape != y.shape:
            residual = self.residual(x)
            residual = self.bn_residual(residual)
        return self.act(y + residual)

class Encoder(nn.Module):
    def __init__(self, g_dim, init_filters=64, n_channel=3):
        super(Encoder, self).__init__()
        self.init_filters = init_filters
        self.g_dim = g_dim
        self.layer1 = EncoderBlock(in_filters=n_channel, out_filters = init_filters, strides=2)
        self.layer2 = EncoderBlock(in_filters=init_filters, out_filters = init_filters, strides=1)
        self.layer3 = EncoderBlock(in_filters=init_filters, out_filters = init_filters * 2, strides=2)
        self.layer4 = EncoderBlock(in_filters=init_filters * 2, out_filters = init_filters * 2, strides=1)
        self.layer5 = EncoderBlock(in_filters=init_filters * 2, out_filters = init_filters * 4, strides=2)
        self.layer6 = EncoderBlock(in_filters=init_filters * 4, out_filters = init_filters * 4, strides=1)
        self.layer7 = EncoderBlock(in_filters=init_filters * 4, out_filters = init_filters * 8, strides=2)
        self.layer8 = EncoderBlock(in_filters=init_filters * 8, out_filters = init_filters * 8, strides=1)
        self.layer9 = nn.Sequential(
            nn.Conv2d(init_filters * 8, g_dim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(g_dim),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        x = self.layer1(x)
        x = self.layer2(x)
        skips.append(x)
        x = self.layer3(x)
        x = self.layer4(x)
        skips.append(x)
        x = self.layer5(x)
        x = self.layer6(x)
        skips.append(x)
        x = self.layer7(x)
        x = self.layer8(x)
        skips.append(x)
        x = self.layer9(x)
        return x.view(-1, self.g_dim), skips

class DecoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters, upsample=False, expand=4):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        self.bn0 = nn.BatchNorm2d(in_filters)
        self.conv1 = nn.Conv2d(in_filters, out_filters * expand, kernel_size=(1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_filters * expand)
        self.conv2 = nn.Conv2d(out_filters * expand, out_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.act = swish()
        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(1, 1), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.se_block = SEBlock(c=out_filters)
        self.residual = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=(1, 1))
        self.bn_residual = nn.BatchNorm2d(out_filters)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def upsample_image(self, x, multiplier=2):
        shape = (x.shape[2] * multiplier,
                 x.shape[3] * multiplier)
        return Resize(size=shape, interpolation=torchvision.transforms.InterpolationMode.NEAREST)(x)

    def forward(self, x):
        if self.upsample:
            x = self.up(x)
        residual = x
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.se_block(x)
        if x.shape != residual.shape:
            residual = self.residual(residual)
            residual = self.bn_residual(residual)
        return self.act(x + residual)

class Decoder(nn.Module):
    def __init__(self, g_dim=512, init_filters=64, skip_type='residual', n_channel=3):
        super(Decoder, self).__init__()
        self.skip_type = skip_type
        self.n_channel = n_channel
        self.g_dim = g_dim
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(g_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            swish()
        )
        if self.skip_type == 'residual':
            self.layer1 = DecoderBlock(in_filters=init_filters, out_filters=init_filters, upsample=True)
        else:
            self.layer1 = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters, upsample=True)
        self.layer2 = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters, upsample=True)
        if self.skip_type == 'residual':
            self.layer3 = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2)
        else:
            self.layer3 = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters * 2)
        self.layer4 = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters * 2, upsample=True)
        if self.skip_type == 'residual':
            self.layer5 = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters * 4)
        else:
            self.layer5 = DecoderBlock(in_filters=init_filters * 8, out_filters=init_filters * 4)
        self.layer6 = DecoderBlock(in_filters=init_filters * 8, out_filters=init_filters * 4, upsample=True)
        if self.skip_type == 'residual':
            self.layer7 = DecoderBlock(in_filters=init_filters * 8, out_filters=init_filters * 8)
        else:
            self.layer7 = DecoderBlock(in_filters=init_filters * 16, out_filters=init_filters * 8)
        self.layer8 = DecoderBlock(in_filters=init_filters * 8, out_filters=init_filters * 8)
        self.conv_out = nn.Conv2d(init_filters, self.n_channel, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        y, skips = x
        y = self.upc1(y.view(-1, self.g_dim, 1, 1))  # 1 -> 4
        y = self.layer8(y)                           # 4 -> 4
        if self.skip_type == 'residual':
            y = y + skips[-1]
        else:
            y = torch.cat([y, skips[-1]], dim=1)
        y = self.layer7(y)
        y = self.layer6(y)                          # 4 -> 8
        if self.skip_type == 'residual':
            y = y + skips[-2]
        else:
            y = torch.cat([y, skips[-2]], dim=1)
        y = self.layer5(y)
        y = self.layer4(y)                          # 8 -> 16
        if self.skip_type == 'residual':
            y = y + skips[-3]
        else:
            y = torch.cat([y, skips[-3]], dim=1)
        y = self.layer3(y)
        y = self.layer2(y)                          # 16 -> 32
        if self.skip_type == 'residual':
            y = y + skips[-4]
        else:
            y = torch.cat([y, skips[-4]], dim=1)
        y = self.layer1(y)
        y = self.conv_out(y)
        y = torch.sigmoid(y)
        return y

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # nn.BatchNorm1d(output_size),
            nn.Tanh())
        self.hidden = self.init_hidden(device)

    def init_hidden(self, device):
        return [(Variable(torch.zeros(self.batch_size, self.hidden_size).to(device)),
                 Variable(torch.zeros(self.batch_size, self.hidden_size).to(device)))
                for _ in range(self.n_layers)]

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)


class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden(device)

    def init_hidden(self, device):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).to(device)),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).to(device))))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.main(input)
        return out