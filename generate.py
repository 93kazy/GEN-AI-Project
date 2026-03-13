import torch
import torchvision
import os
import argparse
import math

from model import Generator, Discriminator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate with Discriminator Driven Latent Sampling.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    print('Model Loading...')
    mnist_dim = 784

    model_G = Generator(g_output_dim=mnist_dim).to(device)
    model_D = Discriminator(d_input_dim=mnist_dim).to(device)

    model_G, model_D = load_model(model_G, model_D, 'checkpoints', device)

    if torch.cuda.device_count() > 1:
        model_G = torch.nn.DataParallel(model_G)
        model_D = torch.nn.DataParallel(model_D)

    model_G.eval()
    model_D.eval()
    print('Models loaded.')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    steps = 1000
    eps = 0.005
    eps_min = 1e-6
    n_samples = 0
    noise_factor = 0.1
    while n_samples < 10000:
        z = torch.randn(args.batch_size, 100).to(device)
        eps0 = eps
        for i in range(steps):
            cos = 0.5 * (1 + math.cos(math.pi * i / steps))
            epsilon = eps_min + (eps0 - eps_min) * cos
            z.requires_grad_(True)
            if z.grad is not None:
              z.grad.detach_()
              z.grad.zero_()

            x = model_G(z)
            d_out = model_D(x)
            d = d_out.squeeze(-1)

            prior_energy = 0.5 * torch.sum(z**2, dim=1)

            energy = prior_energy - d_out.squeeze(-1)
            grad_z = torch.autograd.grad(outputs=energy.sum(), inputs=z)[0]

            with torch.no_grad():
                noise = torch.randn_like(z) * noise_factor

                z = z - (epsilon / 2) * grad_z + math.sqrt(epsilon) * noise

                z = z.detach()

        with torch.no_grad():
            x = model_G(z)
            x = x.reshape(args.batch_size, 1, 28, 28)

            for k in range(x.shape[0]):
                if n_samples < 10000:
                    torchvision.utils.save_image(x[k], os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1
"""
    steps = 1000
    eps = 0.005
    eps_min =  1e-6
    n_samples = 0
    #beta = 0.2
    noise_factor = 0.01
    while n_samples < 10000:
        z = torch.randn(args.batch_size, 100).to(device)
        z_prime = torch.randn(args.batch_size, mnist_dim).to(device)
        eps0 = eps
        for i in range(steps):
            cos =  0.5 * (1 + math.cos(math.pi * i / steps))
            epsilon = eps_min + (eps0 - eps_min) * cos

            z.requires_grad_(True)
            z_prime.requires_grad_(True)

            x_base = model_G(z)
            x = x_base + noise_factor * z_prime

            d_raw = model_D(x).squeeze(-1)
            d = torch.sigmoid(d_raw)

            prior_energy = 0.5 * torch.sum(z**2, dim=1)
            prior_energy_prime = 0.5 * torch.sum(z_prime**2, dim=1)
            #energy = prior_energy + prior_energy_prime - d * beta
            energy = prior_energy + prior_energy_prime - d

            # grad_z = torch.autograd.grad(outputs=energy.sum(), inputs=z)[0]
            grad_z, grad_z_prime = torch.autograd.grad(outputs=energy.sum(), inputs=(z, z_prime))

            with torch.no_grad():
                noise = torch.randn_like(z)
                noise_z_prime = torch.randn_like(z_prime)

                z = z - (epsilon / 2) * grad_z + math.sqrt(epsilon) * noise
                z_prime = z_prime - (epsilon / 2) * grad_z_prime + math.sqrt(epsilon) * noise_z_prime

                #z.clamp_(-0.01 ,0.01)
                z = z.detach()
                z_prime = z_prime.detach()

        with torch.no_grad():
            x = model_G(z)
            x = x.reshape(args.batch_size, 1, 28, 28)

            for k in range(x.shape[0]):
                if n_samples < 10000:
                    torchvision.utils.save_image(
                        x[k],
                        os.path.join('samples', f'{n_samples}.png'),
                        normalize=True,
                        value_range=(-1, 1))
                    n_samples += 1
"""