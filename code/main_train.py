from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import prepare_data, TextBertDataset
from eval.IS.bird.inception_score_bird import compute_IS
from eval.FID.fid_score import compute_FID

from DAMSM import BERT_RNN_ENCODER
from transformers import AutoTokenizer, AutoModel

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD
from torchvision.models import inception_v3
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # or "n"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--evaluation', type=int, help='evaluation', default= 0)
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, dataloader, device, validation=False):
    """
    Generate and save fake images from text embeddings.
    """

    last_epoch = 0
    model_path = f"../models/{cfg.CONFIG_NAME}/checkpoint_nets.pth"

    # Load generator checkpoint if available (unless in validation mode)
    if not validation and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        netG.load_state_dict(checkpoint["netG_state"])
        last_epoch = checkpoint["epoch"]
        netG.eval()
        print("Loading generator checkpoint from epoch:", last_epoch)
    elif not validation:
        print("No checkpoint found, starting without pretrained weights.")

    batch_size = cfg.TRAIN.BATCH_SIZE
    save_dir = f"../images/{cfg.CONFIG_NAME}/test"
    mkdir_p(save_dir)

    counter = 0
    for i in range(1):  # placeholder for cfg.TEXT.CAPTIONS_PER_IMAGE
        for step, data in enumerate(dataloader, 0):
            images, captions, cap_lens, class_ids, keys = prepare_data(data)
            counter += batch_size

            hidden = text_encoder.init_hidden(batch_size)
            _, sent_embs = text_encoder(captions, cap_lens, hidden)
            sent_embs = sent_embs.detach()

            # Generate fake images with noise input
            noise = torch.randn(batch_size, 100, device=device)
            with torch.no_grad():
                fake_imgs = netG(noise, sent_embs)

            # Save each image
            for j, key in enumerate(keys):
                base_path = f"{save_dir}/{key}"
                folder = os.path.dirname(base_path)
                if not os.path.isdir(folder):
                    print("Creating new folder:", folder)
                    mkdir_p(folder)

                img_arr = fake_imgs[j].cpu().numpy()
                img_arr = ((img_arr + 1.0) * 127.5).astype(np.uint8)  # scale [-1, 1] → [0, 255]
                img_arr = np.transpose(img_arr, (1, 2, 0))

                img = Image.fromarray(img_arr)
                img.save(f"{base_path}_{i}.png")

    return last_epoch


def validate(text_encoder, netG,device, writer, epoch):
    dataset = TextBertDataset(cfg.DATA_DIR, 'test',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))
    print(f'Starting generate validation images ...  at {epoch}')    
    sampling(text_encoder, netG, dataloader, device, validation= True)
    
    netG.train()
    
     
    print(f'Starting compute FID & IS ... at {epoch}')
    
    compute_FID(['/home/Srinivas/Text to Image/Hindi/data/CUB-200/CUB_val.npz', 
        '../images/%s/test' % (cfg.CONFIG_NAME)], writer, epoch)
    
    mean, std = compute_IS('../images/%s/test' % (cfg.CONFIG_NAME), writer, epoch)
    final_score = mean / std
    print(f"Final Inception Score: {final_score:.4f}")
    

    
    
#########################################




########################################    
    
  
  
def train(dataloader, netG, netD, text_encoder, optimizerG, optimizerD, start_epoch, batch_size, device, writer):
  

    checkpoint_path = f"../models/{cfg.CONFIG_NAME}/checkpoint_nets.pth"

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        netG.load_state_dict(ckpt['netG_state'])
        netD.load_state_dict(ckpt['netD_state'])
        optimizerG.load_state_dict(ckpt['optimizerG_state'])
        optimizerD.load_state_dict(ckpt['optimizerD_state'])
        start_epoch = ckpt['epoch']
        netG.train()
        netD.train()
        print(f"Resuming training from checkpoint at epoch {start_epoch}")
    else:
        print("No existing checkpoint found, starting new training.")

    # Main training loop
    for epoch in range(start_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1):
        cumulative_d_loss, cumulative_g_loss = 0.0, 0.0

        for step, batch in enumerate(dataloader, start=0):
            images, captions, cap_lens, class_ids, keys = prepare_data(batch)
            hidden_state = text_encoder.init_hidden(batch_size)

            # Extract embeddings
            word_embs, sent_embs = text_encoder(captions, cap_lens, hidden_state)
            word_embs, sent_embs = word_embs.detach(), sent_embs.detach()

            real_imgs = images[0].to(device)
            real_features = netD(real_imgs)

            # Discriminator real & mismatch loss
            real_out = netD.COND_DNET(real_features, sent_embs)
            loss_d_real = torch.nn.ReLU()(1.0 - real_out).mean()

            mismatch_out = netD.COND_DNET(real_features[:batch_size - 1], sent_embs[1:batch_size])
            loss_d_mismatch = torch.nn.ReLU()(1.0 + mismatch_out).mean()

            # Generate fake samples
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = netG(z, sent_embs)

            fake_features = netD(fake_imgs.detach())
            loss_d_fake = netD.COND_DNET(fake_features, sent_embs)
            loss_d_fake = torch.nn.ReLU()(1.0 + loss_d_fake).mean()

            # Combine discriminator loss
            total_d_loss = loss_d_real + (loss_d_fake + loss_d_mismatch) / 2.0
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            total_d_loss.backward()
            optimizerD.step()

            # MA-GP regularization
            interpolated_imgs = real_imgs.data.requires_grad_()
            interpolated_sent = sent_embs.data.requires_grad_()
            interp_features = netD(interpolated_imgs)
            interp_out = netD.COND_DNET(interp_features, interpolated_sent)

            grads = torch.autograd.grad(
                outputs=interp_out,
                inputs=(interpolated_imgs, interpolated_sent),
                grad_outputs=torch.ones_like(interp_out).cuda(),
                retain_graph=True,
                create_graph=True,
                only_inputs=True
            )

            grad_img = grads[0].view(grads[0].size(0), -1)
            grad_sent = grads[1].view(grads[1].size(0), -1)
            grad_concat = torch.cat((grad_img, grad_sent), dim=1)
            grad_norm = torch.sqrt(torch.sum(grad_concat ** 2, dim=1))
            gp_loss = torch.mean((grad_norm) ** 6)
            reg_loss_d = 2.0 * gp_loss

            optimizerD.zero_grad()
            optimizerG.zero_grad()
            reg_loss_d.backward()
            optimizerD.step()

            # Generator loss
            gen_features = netD(fake_imgs)
            gen_out = netD.COND_DNET(gen_features, sent_embs)
            g_loss = -gen_out.mean()

            optimizerG.zero_grad()
            optimizerD.zero_grad()
            g_loss.backward()
            optimizerG.step()

            # Track losses
            cumulative_d_loss += total_d_loss.item() + reg_loss_d.item()
            cumulative_g_loss += g_loss.item()

            print(
                f"[{epoch}/{cfg.TRAIN.MAX_EPOCH}][{step}/{len(dataloader)}] "
                f"D_Loss: {total_d_loss.item():.3f} G_Loss: {g_loss.item():.3f} "
                f"Cumulative_D: {cumulative_d_loss:.3f} Cumulative_G: {cumulative_g_loss:.3f}"
            )

        # Save generated samples
        vutils.save_image(
            fake_imgs.data,
            f"../images/{cfg.CONFIG_NAME}/fakes/fake_samples_epoch_{epoch:03d}.png",
            normalize=True
        )

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'netG_state': netG.state_dict(),
            'optimizerG_state': optimizerG.state_dict(),
            'netD_state': netD.state_dict(),
            'optimizerD_state': optimizerD.state_dict()
        }, checkpoint_path)

        # Log metrics
        writer.add_scalar('D_Loss/train', cumulative_d_loss, epoch)
        writer.add_scalar('G_Loss/train', cumulative_g_loss, epoch)

        # Optional early exit
        if epoch % 50 == 0:
            return epoch

    return cfg.TRAIN.MAX_EPOCH




if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    
    cfg.B_VALIDATION = bool(args.evaluation)

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextBertDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:     
        dataset = TextBertDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    
    text_encoder = BERT_RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    state_dict.pop('encoder.embeddings.word_embeddings.weight', None)
    text_encoder.load_state_dict(state_dict, strict=False)
    
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    state_epoch=0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))  


    if cfg.B_VALIDATION:
        state_epoch = sampling(text_encoder, netG, dataloader,device)  # generate images for the whole valid dataset
        print('state_epoch:  %d'%(state_epoch))
    else:
        writer = SummaryWriter(f"tensorboards/{cfg.CONFIG_NAME}/ADGAN_train")
        epoch = train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD, state_epoch,batch_size,device, writer)
        validate(text_encoder, netG, device, writer, epoch)


        
