import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexConvTranspose2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VariationalEncoder(nn.Module):

    def __init__(self, latent_dim):
        super(VariationalEncoder, self).__init__()

        # encoder
        self.conv1 = ComplexConv2d(in_channels=1, out_channels=512, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = ComplexConv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = ComplexConv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = ComplexConv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = ComplexConv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=(1, 1))

        # distribution parameters
        self.mu = ComplexLinear(32 * 31 * 3, latent_dim)
        self.sigma = ComplexLinear(32 * 31 * 3, latent_dim)
        self.delta = ComplexLinear(32 * 31 * 3, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def compute_reparam_trick(self, mu, sigma, delta):
        numerator = sigma ** 2 - torch.abs(delta) ** 2
        denominator = 2 * sigma + 2 * torch.real(delta)
        numerator_sign = torch.sign(numerator.detach())
        denominator_sign = torch.sign(denominator.detach())
        numerator_sign = (numerator_sign - 1) / 2 * -1j + (
                (numerator_sign + 1) / 2)  # That's how we convert -1 to 1j and keep 1 as 1
        denominator_sign = (denominator_sign - 1) / 2 * -1j + ((denominator_sign + 1) / 2)
        numerator = numerator_sign * torch.sqrt(torch.abs(numerator))
        denominator = denominator_sign * torch.sqrt(torch.abs(denominator)) + 1e-8
        kx = (sigma + delta) / denominator
        ky = 1j * numerator / denominator
        h = mu + kx * self.N.sample(mu.shape) + ky * self.N.sample(mu.shape)
        return h

    def compute_complex_kl(self, mu, sigma, delta):
        logarithm = sigma ** 2 - torch.abs(delta) ** 2
        self.kl = torch.real((torch.conj(mu) * mu).sum(axis=1)) + torch.abs(
            (sigma - 1 - (torch.log(torch.abs(logarithm) + 1e-8)) / 2)).sum(axis=1)

    def forward(self, x):
        x = x.to(device)
        x = complex_relu(self.conv1(x))
        x = complex_relu(self.conv2(x))
        x = complex_relu(self.conv3(x))
        x = complex_relu(self.conv4(x))
        x = complex_relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.mu(x)
        sigma = torch.exp(torch.real(self.sigma(x)))
        delta = self.delta(x) * 1e-8
        h = self.compute_reparam_trick(mu, sigma, delta)
        self.compute_complex_kl(mu, sigma, delta)
        return h


class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.lim_linear1 = ComplexLinear(latent_dims, 128)
        self.lim_linear2 = ComplexLinear(128, 32 * 31 * 3)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 31, 3))

        self.dec1 = ComplexConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.dec2 = ComplexConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.dec3 = ComplexConvTranspose2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                                           output_padding=1)
        self.dec4 = ComplexConvTranspose2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2),
                                           output_padding=1)
        self.dec5 = ComplexConvTranspose2d(in_channels=512, out_channels=1, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=1)

    def forward(self, x):
        x = complex_relu(self.lim_linear1(x))
        x = complex_relu(self.lim_linear2(x))
        x = self.unflatten(x)
        x = complex_relu(self.dec1(x))
        x = complex_relu(self.dec2(x))
        x = complex_relu(self.dec3(x))
        x = complex_relu(self.dec4(x))
        x = self.dec5(x)
        x = torch.tanh(x)
        return x


class CVAE(nn.Module):
    def __init__(self, latent_dims):
        super(CVAE, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
