import torch
import torch.nn as nn
from torch.autograd import Variable

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim), [h1, h2, h3, h4, h5]


class decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.upc5 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 64 x 64
        self.upc6 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 128 x 128
                )

    def forward(self, input, skips0, skips1, skips2, skips3, skip4):
        # vec, skip = input
        vec = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip4], 1))
        d3 = self.upc3(torch.cat([d2, skips3], 1))
        d4 = self.upc4(torch.cat([d3, skips2], 1))
        d5 = self.upc5(torch.cat([d4, skips1], 1))
        output = self.upc6(torch.cat([d5, skips0], 1))
        return output

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
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
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size)),
                           Variable(torch.zeros(self.batch_size, self.hidden_size))))
        return hidden


    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        return self.output(h_in)