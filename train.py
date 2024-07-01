import torch
import torch.optim as optim
from torchvision.utils import make_grid, save_image

import os
from tqdm import trange, tqdm

from score.both import get_inception_and_fid_score
import json

import wandb

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))
        
def evaluate(sampler, model, args, device):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, 3, args.img_size, args.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, args.fid_cache, num_images=args.num_images,
        use_torch=args.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images
        

def train(args, net_model, ema_model, trainer, net_sampler, ema_sampler, device, dl, datalooper):
    warmup_lr = lambda step: min(step, args.warmup) / args.warmup
    optimizer = optim.Adam(net_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    if not os.path.exists(os.path.join(args.logdir, 'sample')):
        os.makedirs(os.path.join(args.logdir, 'sample'))
    x_T = torch.randn(args.sample_size, 3, args.img_size, args.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dl))[0][:args.sample_size]) + 1) / 2

    # # backup all arguments
    # with open(os.path.join(args.logdir, "flagfile.txt"), 'w') as f:
    #     f.write(args)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    #training
    for step in tqdm(range(args.total_steps), dynamic_ncols=True):
        optimizer.zero_grad()
        x_0 = next(datalooper).to(device)
        loss = trainer(x_0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net_model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema(net_model, ema_model, args.ema_decay)
        wandb.log({"Training loss": loss})

        # sample
        if args.sample_step > 0 and step % args.sample_step == 0:
            net_model.eval()
            with torch.no_grad():
                x_0 = ema_sampler(x_T.to(device))
                grid = (make_grid(x_0) + 1) / 2
                path = os.path.join(
                    args.logdir, 'sample', '%d.png' % step)
                save_image(grid, path)
            net_model.train()
            wandb.log({
                'images': wandb.Image(grid)
            })            

        # save
        if args.save_step > 0 and step % args.save_step == 0:
            ckpt = {
                'net_model': net_model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'sched': scheduler.state_dict(),
                'optim': optimizer.state_dict(),
                'step': step,
                'x_T': x_T,
            }
            torch.save(ckpt, os.path.join(args.logdir, 'ckpt.pt'))

        # evaluate
        if args.eval_step > 0 and step % args.eval_step == 0:
            net_IS, net_FID, _ = evaluate(net_sampler, net_model, args, device)
            ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model, args, device)
            metrics = {
                'IS': net_IS[0],
                'IS_std': net_IS[1],
                'FID': net_FID,
                'IS_EMA': ema_IS[0],
                'IS_std_EMA': ema_IS[1],
                'FID_EMA': ema_FID
            }
            print(metrics)
            wandb.log(metrics)


            with open(os.path.join(args.logdir, 'eval.txt'), 'a') as f:
                metrics['step'] = step
                f.write(json.dumps(metrics) + "\n")