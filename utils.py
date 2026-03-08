import torch
import os



def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()
    M = x.shape[0]
    z = torch.randn(M, 100, device=device)

    steps = 10
    epsilon = 0.01
    for i in range(steps):
        z.requires_grad_(True)
        
        G_z = G(z)
        D_G_z = D(G_z).squeeze() 
 
        prior_energy = 0.5 * torch.sum(z**2, dim=1)
        energy = prior_energy - D_G_z
        
        grad_z = torch.autograd.grad(outputs=energy.sum(), inputs=z)[0]
        noise = torch.randn_like(z)
        
        with torch.no_grad():
            z = z - (epsilon / 2) * grad_z + math.sqrt(epsilon) * noise
            z = z.detach()
            
    x_real = x.to(device)
    x_fake_mcmc = G(z).detach()
    
    D_real = D(x_real).mean()
    D_fake = D(x_fake_mcmc).mean()
    
    D_loss = D_fake - D_real
    
    D_loss.backward()
    D_optimizer.step()
    
    for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)
        
    return D_loss.item()

    
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
    M = x.shape[0]
    z = torch.randn(M, 100, device=device)
    x_fake = G(z)
    D_fake = D(x_fake).mean()
    G_loss = -D_fake
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.item()
    
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




