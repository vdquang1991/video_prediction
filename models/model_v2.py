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

class Activation(nn.Module):
    def __init__(self, activation_type, negative_slope=0.2, inplace=True):
        super(Activation, self).__init__()
        self.activation_type = activation_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        if self.activation_type == 'relu':
            return nn.ReLU(inplace=self.inplace)(x)
        if self.activation_type == 'leaky':
            return nn.LeakyReLU(negative_slope=self.negative_slope, inplace=self.inplace)(x)
        if self.activation_type == 'elu':
            return nn.ELU(inplace=self.inplace)(x)
        if self.activation_type == 'swish':
            return swish()(x)

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
    def __init__(self, in_filters, out_filters, strides=1, activation_type='leaky'):
        super(EncoderBlock, self).__init__()
        self.strides = strides
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.act = Activation(activation_type)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.se_block = SEBlock(c=out_filters)
        self.residual = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides)
        self.bn_residual = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.se_block(y)
        if x.shape != y.shape:
            residual = self.residual(x)
            residual = self.bn_residual(residual)
        return self.act(y + residual)

class Encoder(nn.Module):
    def __init__(self, g_dim, init_filters=64, n_channel=3, activation_type='leaky'):
        super(Encoder, self).__init__()
        self.init_filters = init_filters
        self.g_dim = g_dim
        self.layer1 = EncoderBlock(in_filters=n_channel, out_filters=init_filters, strides=2, activation_type=activation_type)
        self.layer2 = EncoderBlock(in_filters=init_filters, out_filters=init_filters, strides=1, activation_type=activation_type)
        self.layer3 = EncoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2, strides=2, activation_type=activation_type)
        self.layer4 = EncoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2, strides=1, activation_type=activation_type)
        self.layer1_diff = EncoderBlock(in_filters=n_channel, out_filters=init_filters, strides=2, activation_type=activation_type)
        self.layer2_diff = EncoderBlock(in_filters=init_filters, out_filters=init_filters, strides=1, activation_type=activation_type)
        self.layer3_diff = EncoderBlock(in_filters=init_filters, out_filters=init_filters * 2, strides=2, activation_type=activation_type)
        self.layer4_diff = EncoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2, strides=1, activation_type=activation_type)
        self.conv_out_rgb = nn.Sequential(
            nn.Conv2d(init_filters * 4, self.g_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.g_dim),
            nn.Tanh())
        self.conv_out_diff = nn.Sequential(
            nn.Conv2d(init_filters * 2, self.g_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.g_dim),
            nn.Tanh())

    def forward(self, x_rgb, x_diff):
        skips = []
        x_rgb = self.layer1(x_rgb)              # 64x32x32
        x_rgb = self.layer2(x_rgb)              # 64x32x32
        x_diff = self.layer1_diff(x_diff)       # 64x32x32
        x_diff = self.layer2_diff(x_diff)       # 64x32x32
        x_rgb = torch.cat([x_rgb, x_diff], dim=1)   # 128x32x32
        skips.append(x_rgb)
        x_rgb = self.layer3(x_rgb)              # 128x16x16
        x_rgb = self.layer4(x_rgb)              # 128x16x16
        x_diff = self.layer3_diff(x_diff)       # 128x16x16
        x_diff = self.layer4_diff(x_diff)       # 128x16x16
        x_rgb = torch.cat([x_rgb, x_diff], dim=1)   # 256x16x16
        skips.append(x_rgb)
        x_rgb = self.conv_out_rgb(x_rgb)                # g_dimx16x16
        x_diff = self.conv_out_diff(x_diff)             # g_dimx16x16
        x = torch.cat([x_rgb, x_diff], dim=1)       # g_dimx2 x 32 x32
        return x, skips


class DecoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters, upsample=False, activation_type='leaky'):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.act = Activation(activation_type)
        self.se_block = SEBlock(c=out_filters)
        self.residual = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=(1, 1))
        self.bn_residual = nn.BatchNorm2d(out_filters)

    def upsample_image(self, x, multiplier=2):
        shape = (x.shape[2] * multiplier,
                 x.shape[3] * multiplier)
        return Resize(size=shape, interpolation=torchvision.transforms.InterpolationMode.NEAREST)(x)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_image(x, multiplier=2)
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se_block(x)
        if x.shape != residual.shape:
            residual = self.residual(residual)
            residual = self.bn_residual(residual)
        return self.act(x + residual)

class Decoder(nn.Module):
    def __init__(self, g_dim=512, init_filters=64, skip_type='residual', n_channel=3, activation_type='leaky'):
        super(Decoder, self).__init__()
        self.skip_type = skip_type
        self.n_channel = n_channel
        self.g_dim = g_dim
        self.init_filters = init_filters

        if self.skip_type == 'residual':
            self.layer1_rgb = DecoderBlock(in_filters=init_filters, out_filters=init_filters, upsample=True, activation_type=activation_type)
            self.layer1_diff = DecoderBlock(in_filters=init_filters, out_filters=init_filters, upsample=True, activation_type=activation_type)
        else:
            self.layer1_rgb = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters, upsample=True, activation_type=activation_type)
            self.layer1_diff = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters, upsample=True, activation_type=activation_type)
        self.layer2_rgb = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters, activation_type=activation_type)
        self.layer2_diff = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters, activation_type=activation_type)
        if self.skip_type == 'residual':
            self.layer3_rgb = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2, upsample=True, activation_type=activation_type)
            self.layer3_diff = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2, upsample=True, activation_type=activation_type)
        else:
            self.layer3_rgb = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters * 2, upsample=True, activation_type=activation_type)
            self.layer3_diff = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters * 2, upsample=True, activation_type=activation_type)
        self.layer4_rgb = DecoderBlock(in_filters=init_filters * 4, out_filters=init_filters * 2, activation_type=activation_type)
        self.layer4_diff = DecoderBlock(in_filters=init_filters * 2, out_filters=init_filters * 2, activation_type=activation_type)

        self.conv_out_rgb = nn.Conv2d(init_filters, self.n_channel, kernel_size=7, stride=1, padding=3)
        self.conv_out_diff = nn.Conv2d(init_filters, self.n_channel, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x, skips = x
        _, x_diff = torch.split(x, self.g_dim, dim=1)                   # 128x16x16 and 128x16x16
        x_rgb = self.layer4_rgb(x)                                      # 128x16x16
        x_diff = self.layer4_diff(x_diff)                                   # 128x16x16
        skip_rgb, skip_diff = torch.split(skips[-1], self.init_filters * 2, dim=1)     # 128x16x16 and 128x16x16
        if self.skip_type == 'residual':
            x_rgb = x_rgb + skip_rgb                                        # 128x16x16
            x_diff = x_diff + skip_diff                                     # 128x16x16
        else:
            x_rgb = torch.cat([x_rgb, skip_rgb], dim=1)                     # 256x16x16
            x_diff = torch.cat([x_diff, skip_diff], dim=1)                  # 256x16x16
        x_rgb = self.layer3_rgb(x_rgb)                                      # 128x32x32
        x_diff = self.layer3_diff(x_diff)                                   # 128x32x32
        x_rgb = torch.cat([x_rgb, x_diff], dim=1)                           # 256x32x32
        x_rgb = self.layer2_rgb(x_rgb)                                      # 64x32x32
        x_diff = self.layer2_diff(x_diff)                                   # 64x32x32
        skip_rgb, skip_diff = torch.split(skips[-2], self.init_filters, dim=1)  # 64x32x32 and 64x32x32
        if self.skip_type == 'residual':
            x_rgb = x_rgb + skip_rgb                                        # 64x32x32
            x_diff = x_diff + skip_diff                                     # 64x32x32
        else:
            x_rgb = torch.cat([x_rgb, skip_rgb], dim=1)                     # 128x32x32
            x_diff = torch.cat([x_diff, skip_diff], dim=1)                  # 128x32x32
        x_rgb = self.layer1_rgb(x_rgb)                                      # 64x64x64
        x_diff = self.layer1_diff(x_diff)                                   # 64x64x64
        x_rgb = self.conv_out_rgb(x_rgb)                                    # n_channelx64x64
        x_diff = self.conv_out_diff(x_diff)                                 # n_channelx64x64
        x_rgb = torch.sigmoid(x_rgb)
        x_diff = torch.sigmoid(x_diff)
        return x_rgb, x_diff


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class conv_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, image_size, device = torch.device('cpu')):
        super(conv_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.embed = nn.Conv2d(self.input_size, hidden_size, 3, 1, 1)
        self.lstm = nn.ModuleList([ConvLSTMCell(hidden_size, hidden_size, (3, 3), True) for _ in range(self.n_layers)])
        self.hidden = self.init_hidden()
        self.output = nn.Sequential(
            nn.Conv2d(hidden_size, output_size, 3, 1, 1),
            nn.Tanh())

    def init_hidden(self, device=torch.device('cpu')):
        hidden = []
        for i in range(self.n_layers):
            hidden.append(self.lstm[i].init_hidden(self.batch_size, self.image_size))
        return hidden

    def forward(self, inp):
        h_in = self.embed(inp)
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        h_in = self.output(h_in)
        return h_in

class gaussian_conv_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, image_size, device=torch.device('cpu')):
        super(gaussian_conv_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.embed = nn.Conv2d(input_size, hidden_size, 3, 1, 1)
        self.lstm = nn.ModuleList([ConvLSTMCell(input_size, hidden_size, (3, 3), True) for i in range(self.n_layers)])
        self.mu_net = nn.Conv2d(hidden_size, output_size, 3, 1, 1)
        self.logvar_net = nn.Conv2d(hidden_size, output_size, 3, 1, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self, device=torch.device('cpu')):
        return [self.lstm[i].init_hidden(self.batch_size, self.image_size) for i in range(self.n_layers)]

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        h_in = input
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar