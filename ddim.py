import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    if isinstance(t, int):
        t = torch.tensor([t], dtype=torch.long, device=v.device)
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DDIMTrainer(nn.Module):
    '''
    train progress for DDIM
    DDIM trainer is same as DDPM
    see paper
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

class DDIMSampler(nn.Module):
    '''
    sampling progress of DDIM
    See DDPM paper Algorithm 2 & DDIM Sampling method
    DDIM using accelerated sampling
    steps : Sampling steps, if steps=1, sampling all steps
    method : linear or quadratic
    eta : 0 = DDIM, 1 = DDPM
    only_return_x_0 : save image only x_0 or all image in sampling steps
    interval : if only_return_x_0 = False, deciding the interval to save the intermediate process pictures
    '''
    def __init__(self, model, beta_1 = 1e-4, beta_T = 0.02, T = 1000, img_size=32, steps= 1, method='linear', eta=0.0, only_return_x_0=True, interval=1):
        # assert mean_type in ['xprev' 'xstart', 'epsilon']
        # assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.steps = steps
        self.method = method
        self.eta = eta
        self.only_return_x_0 = only_return_x_0
        self.interval = interval

        # beta_1 to beta_T using linear scheduling
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer('alpha_bar', torch.cumprod(1. - self.betas, dim=0))

    def sample_one_step(self, x_t, time_step, prev_time_step, eta):
        device = x_t.device

        # set time step
        t = torch.tensor([time_step], dtype=torch.long, device=device)
        prev_t = torch.tensor([prev_time_step], dtype=torch.long, device=device)

        alpha_t_bar = extract(self.alpha_bar, t, x_t.shape)
        prev_alpha_t_bar = extract(self.alpha_bar, prev_t, x_t.shape)

        # predict model
        epsilon = self.model(x_t, t)        

        #sigma setting
        # if eta=1 DDPM, eta=0 DDIM
        sigma = eta * torch.sqrt((1 - prev_alpha_t_bar) / (1 - alpha_t_bar) * (1 - alpha_t_bar / prev_alpha_t_bar))
        epsilon_t = torch.randn_like(x_t)

        # sampling
        x_t_minus_one = (torch.sqrt(prev_alpha_t_bar / alpha_t_bar) * (x_t - torch.sqrt(1 - alpha_t_bar) * epsilon)
                         + torch.sqrt(1 - prev_alpha_t_bar - sigma**2) * epsilon + sigma*epsilon_t)
        
        return x_t_minus_one

    def forward(self, x_t):
        '''
        x_t : standard gaussian
        '''
        if self.method == "linear":
            accelerate = self.T // self.steps
            time_step = np.asarray(list(range(0, self.T, accelerate)))
        elif self.method == "quadratic":
            time_step = (np.linspace(0, np.sqrt(self.T * 0.8), self.steps) ** 2).astype(np.int)
        else:
            raise NotImplementedError(f"sampling method {self.method} is not implemented!")

        # add one to get final alpha values right
        time_step = time_step + 1
        time_step_prev = np.concatenate([[0], time_step[:-1]])

        # Sampling list
        x = [x_t]
        for i in tqdm(reversed(range(self.steps)), desc='ddpm reversed_process', dynamic_ncols=True):
            x_t = self.sample_one_step(x_t, time_step[i], time_step_prev[i], self.eta)

            if not self.only_return_x_0 and ((self.steps - i) % self.interval ==0 or i == 0):
                x.append(torch.clip(x_t, -1.0, 1.0))
        
        x_0 = x_t
        if self.only_return_x_0:
            return torch.clip(x_0, -1, 1)
        return torch.stack(x, dim=1)