import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    if isinstance(t, int):
        t = torch.tensor([t], dtype=torch.long, device=v.device)
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DDPMTrainer(nn.Module):
    '''
    train progress for DDPM
    DDPM Trainer use linear scheduling for beta
    \beta_1 = 10^{-4}, \beta_T = 0.02 in usual
    using register_buffer for alpha_bar
    '''
    def __init__(self, model, beta_1 = 1e-4, beta_T = 0.02, T = 1000):
        super().__init__()

        # Model would be UNet backbone
        self.model = model
        self.T = T

        # beta_1 to beta_T using linear scheduling
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer('alpha_bar', torch.cumprod(1. - self.betas, dim=0))

    
    def forward(self, x_0):
        '''
        using mse_losses
        \epsilon - \epsilon_\theta(\sqrt{\bar_\alpha_t}x_0 + \sqrt{1 - \bar_\alpha_t}\epsion, t)
        \epsilon_\theta is our model
        see DDPM paper Algorithm 1
        x_0 : (B, H, W, 3)
        t : (B, )
        epsilon : (B, H, W, 3)
        '''

        # t ~ Uniform({1, ... , T})
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # \epsilon ~ \Mathcal N(0, I)
        epsilon = torch.randn_like(x_0)

        x_t = (extract(torch.sqrt(self.alpha_bar), t, x_0.shape) * x_0 +
               extract(torch.sqrt(1. - self.alpha_bar), t, x_0.shape) * epsilon)
        
        loss = F.mse_loss(self.model(x_t, t), epsilon, reduction='none')
        return loss

class DDPMSampler(nn.Module):
    '''
    sampling progress of DDPM
    See DDPM paper Algorithm 2
    '''
    def __init__(self, model, beta_1 = 1e-4, beta_T = 0.02, T = 1000, img_size=32):
        # assert mean_type in ['xprev' 'xstart', 'epsilon']
        # assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size

        # beta_1 to beta_T using linear scheduling
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer('alpha_bar', torch.cumprod(1. - self.betas, dim=0))

    def forward(self, x_T):
        # x_T ~ Normal(0, I)
        x_t = x_T
        for t in tqdm(reversed(range(self.T)), desc='ddpm reversed_process', dynamic_ncols=True):
            if t > 1:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros(x_t.shape)
            z = z.to(x_t.device)
            t = torch.tensor([t], dtype=torch.long, device=x_t.device)
            x_t = ((1/extract(torch.sqrt(1. - self.betas), t, x_t.shape)) * 
                   (x_t - (extract(self.betas, t, x_t.shape) / torch.sqrt(1 - extract(self.alpha_bar, t, x_t.shape)))*self.model(x_t, t))
                   + z*extract(torch.sqrt(self.betas), t, x_t.shape))
        
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
