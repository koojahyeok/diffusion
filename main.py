import copy
import torch
import argparse
import os
from tqdm import trange

from torchvision.datasets import CIFAR10
from torchvision import transforms

from score.both import get_inception_and_fid_score
from torchvision.utils import save_image

from ddpm import DDPMSampler, DDPMTrainer
from model import UNet
import train

def parse_args():
    parser = argparse.ArgumentParser()
    # basic setup
    parser.add_argument('--train', default=True, type=eval, help='train mode or validation model')
    parser.add_argument('--dataset', default='cifar10', type=str, help='choose dataset')

    # UNet setting
    parser.add_argument('--ch', default=128, type=int, help='base channel of UNet')
    parser.add_argument('--ch_mult', default=[1, 2, 2, 2], type=int, help='channel multipler')
    parser.add_argument('--attn', default=[1], type=int, help='add attention to these levels')
    parser.add_argument('--num_res_blocks', default=2, type=int, help='# resblock in each level')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of resblock')

    # Diffusion setting
    parser.add_argument('--beta_1', default=1e-4, type=float, help='start beta value')
    parser.add_argument('--beta_T', default=0.02, type=float, help='end beta value')
    parser.add_argument('--T', default=1000, type=int, help='total diffusion steps')

    # this two is for DDIM
    # parser.add_argument('mean_type', default='epsilon', type=str, help='predict variable')
    # parser.add_argument('var_type', default='fixedlarge', type=str, help='variance type')

    # Training setting
    parser.add_argument('--lr', default=2e-4, type=float, help='target learning rate')
    parser.add_argument('--grad_clip', default=1., type=float, help="gradient norm clipping")
    parser.add_argument('--total_steps', default=800000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=32, type=int, help='image size')
    parser.add_argument('--warmup', default=5000, type=int, help='learning rate warmup')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers of dataloader')
    parser.add_argument('--ema_decay', default=0.9999, type=float, help="ema decay rate")
    parser.add_argument('--parallel', default=False, type=eval, help='multi gpu training')

    # Logging & Sampling
    parser.add_argument('--logdir', default='./logs/DDPM_CIFAR10_EPS', type=str, help='log directory')
    parser.add_argument('--sample_size', default=64, type=int, help="sampling size of images")
    parser.add_argument('--sample_step', default=1000, type=int, help='frequency of sampling')

    # Evaluation
    parser.add_argument('--save_step', default=5000, type=int, help='frequency of saving checkpoints, 0 to disable during training')
    parser.add_argument('--eval_step', default=0, type=int, help='frequency of evaluating model, 0 to disable during training')
    parser.add_argument('--num_images', default=50000, type=int, help='the number of generated images for evaluation')
    parser.add_argument('--fid_use_torch', default=False, type=eval, help='calculate IS and FID on gpu')
    parser.add_argument('--fid_cache', default='./stats/cifar10.train.npz',type=str, help='FID cache')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu_number')
    return parser.parse_args()



def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

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

def main(args, device):
    if args.dataset=='cifar10':
        # dataset
        dataset = CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True)
        datalooper = infiniteloop(dataloader)

    if args.train == True:
        # model setup
        net_model = UNet(
            T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
            num_res_blocks=args.num_res_blocks, dropout=args.dropout)
        ema_model = copy.deepcopy(net_model)
        trainer = DDPMTrainer(
            net_model, args.beta_1, args.beta_T, args.T).to(device)
        net_sampler = DDPMSampler(
            net_model, args.beta_1, args.beta_T, args.T, args.img_size,).to(device)
        ema_sampler = DDPMSampler(
            ema_model, args.beta_1, args.beta_T, args.T, args.img_size,).to(device)
        if args.parallel:
            trainer = torch.nn.DataParallel(trainer)
            net_sampler = torch.nn.DataParallel(net_sampler)
            ema_sampler = torch.nn.DataParallel(ema_sampler)

        train.train(args,net_model, ema_model, trainer, net_sampler, ema_sampler, device, dataloader, datalooper)
    else:
        #validation
        # model setup
        model = UNet(
            T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
            num_res_blocks=args.num_res_blocks, dropout=args.dropout)
        sampler = DDPMSampler(
            model, args.beta_1, args.beta_T, args.T, img_size=args.img_size).to(device)
        if args.parallel:
            sampler = torch.nn.DataParallel(sampler)

        # load model and evaluate
        ckpt = torch.load(os.path.join(args.logdir, 'ckpt.pt'))

        model.load_state_dict(ckpt['ema_model'])
        (IS, IS_std), FID, samples = evaluate(sampler, model)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(args.logdir, 'samples_ema.png'),
            nrow=16)




if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}')
    main(args, device)