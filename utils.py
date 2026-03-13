import torch
import torch.autograd as autograd
import os
import math


def D_train(x, G, D, D_optimizer, device):
    # =======================Train the discriminator=======================#
    lambda_gp = 10
    D.zero_grad()

    real_samples = x.to(device)
    real_validity = D(real_samples)

    z = torch.randn(real_samples.size(0), 100, device=device)
    fake_samples = G(z).detach()
    fake_validity = D(fake_samples)

    gradient_penalty = compute_gradient_penalty(D, real_samples, fake_samples, device)

    d_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gradient_penalty

    d_loss.backward()
    D_optimizer.step()

    return d_loss.item()


def G_train(x, G, D, G_optimizer, device):
    # =======================Train the generator=======================#

    G.zero_grad()

    z = torch.randn(x.size(0), 100, device=device)
    fake_samples = G(z)

    fake_validity = D(fake_samples)

    g_loss = -fake_validity.mean()

    g_loss.backward()
    G_optimizer.step()

    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)

    return g_loss.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))


def load_model(G, D, folder, device):
    G_ckpt_path = os.path.join(folder, 'G.pth')
    D_cpkt_path = os.path.join(folder, 'D.pth')
    G_ckpt = torch.load(G_ckpt_path, map_location=device)
    D_ckpt = torch.load(D_cpkt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in G_ckpt.items()})
    D.load_state_dict({k.replace('module.', ''): v for k, v in D_ckpt.items()})

    return G, D


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).expand_as(real_samples).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)

    fake = torch.ones_like(d_interpolates).to(device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



