import torch
import torch.autograd as autograd
import os
import math



def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
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

    
    """D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()"""


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#

    G.zero_grad()

    z = torch.randn(x.size(0), 100, device=device)
    fake_samples = G(z)

    fake_validity = D(fake_samples)

    g_loss = -fake_validity.mean()

    g_loss.backward()
    G_optimizer.step()

    return g_loss.item()
    
    """G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()"""



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G,D, folder, device):
    G_ckpt_path = os.path.join(folder,'G.pth')
    D_cpkt_path = os.path.join(folder,'D.pth')
    G_ckpt = torch.load(G_ckpt_path, map_location=device)
    D_ckpt = torch.load(D_cpkt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in G_ckpt.items()})
    D.load_state_dict({k.replace('module.', ''): v for k, v in D_ckpt.items()})

    return G,D



def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).expand_as(real_samples).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = D(interpolates)
    
    fake = torch.ones(real_samples.size(0), 1).to(device)

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




