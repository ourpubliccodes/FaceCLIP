from __future__ import print_function

import random

from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from PIL import Image
from tqdm import tqdm

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from VGGFeatureLoss import VGGNet

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss

import os
import time
import numpy as np
import sys
import clip
import logging
import matplotlib.pyplot as plt


def get_log(file_name):
    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever
    formatter = logging.Formatter('%(asctime)s\n%(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger

logger = get_log('../output/log.txt')


def display_loss(data_dir, max_epoch):
    # data_dir = "/home/prmi/FWW/ControlGAN-master/log.txt"
    D_Loss_list = []
    # Train_Accuracy_list = []
    G_Loss_list = []
    # Valid_Accuracy_list = []
    f1 = open(data_dir, 'r')
    data = []
    # 把训练结果输出到result.txt里，比较笨的办法，按字节位数去取数字结果
    for line in f1:
        if (line.find('Loss_D') >= 0):
            D_Loss_list.append(line.split()[1])
        if (line.find('Loss_G') >= 0):
            G_Loss_list.append(line.split()[3])
    f1.close()
    # 迭代了30次，所以x的取值范围为(0，30)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(0, max_epoch)
    x2 = range(0, max_epoch)
    y2 = D_Loss_list
    y4 = G_Loss_list
    plt.subplot(3, 1, 1)
    plt.plot(x1, y2, '.-', label="Loss_D")
    plt.ylabel('D Loss')
    plt.legend(loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(x1, y4, '.-', label="Loss_G", color='r')
    plt.ylabel('G Loss')
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    plt.plot(x2, y2, '.-', label="Loss_D")
    plt.plot(x2, y4, '.-', label="Loss_G", color='r')
    plt.xlabel('D&G loss vs. epoches')
    plt.ylabel('Train Loss')
    plt.legend(loc='best')
    ## 调整每个子图间的间距，hspace调整上下间距，wspace调整左右间距
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    # plt.show()
    plt.savefig('train_loss.png')  # 保存图片


class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        # RN50 or ViT-B/32
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=28)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, dataset):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            self.output_dir = output_dir

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.dataset = dataset
        self.num_batches = len(self.data_loader)

        self.clip_loss = CLIPLoss()

    def build_models(self):
        if cfg.TRAIN.NET_E == '':
            # print('Error: no pretrained text-image encoders')
            logger.info('Error: no pretrained text-image encoders')
            return

        # vgg16 network
        style_loss = VGGNet()

        for p in style_loss.parameters():
            p.requires_grad = False

        # print("Load the style loss model")
        logger.info("Load the style loss model")
        style_loss.eval()

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        # logger.info('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        # logger.info('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
        netG.apply(weights_init)

        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            # print('Load G from: ', cfg.TRAIN.NET_G)
            logger.info('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    # print('Load D from: ', Dname)
                    logger.info('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            style_loss = style_loss.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()

        return [text_encoder, image_encoder, netG, netsD, epoch, style_loss]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(), '%s/netD%d.pth' % (self.model_dir, i))
        # print('Save G/Ds models.')
        logger.info('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png' % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ =  build_super_images(fake_imgs[i].detach().cpu(),
                                         captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png' % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch, style_loss = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                # wrong word and sentence embeddings
                w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                w_words_embs, w_sent_emb = w_words_embs.detach(), w_sent_emb.detach()

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                ############## CLIP encoder ###########
                ixtoword = self.data_loader.dataset.ixtoword
                captions_list = captions.tolist()
                sec_list = []
                for sec_item in captions_list:
                    temp = ''
                    for index in sec_item:
                        if index != 0:
                            temp = temp + ixtoword[index] + ' '
                    sec_list.append(temp)
                #######################################

                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels,
                                              words_embs, cap_lens, image_encoder, class_ids, w_words_embs, 
                                              wrong_caps_len, wrong_cls_id)

                    loss_clip = 0.0
                    for index_sec in range(len(sec_list)):
                        text_input = torch.cat([clip.tokenize(sec_list[index_sec])]).cuda()
                        text_feature = self.clip_loss.model.encode_text(text_input)
                        tensor_img_to_PIL = ToPILImage()(fake_imgs[i][index_sec])
                        image_input = self.clip_loss.preprocess(tensor_img_to_PIL).unsqueeze(0).to("cuda")
                        image_feature = self.clip_loss.model.encode_image(image_input)  # [B, 512]
                        sim = F.cosine_similarity(text_feature, image_feature)  # 相似度，越大越好
                        loss_clip = loss_clip + (1-sim)
                    loss_clip = (loss_clip/(batch_size*1.0))[0]
                    errD += loss_clip

                    # backward and update parameters
                    errD.backward(retain_graph=True)
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD)
                D_logs += 'clip_loss: %.2f ' % loss_clip

                step += 1
                gen_iterations += 1

                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids, style_loss, imgs)
                kl_loss = KL_loss(mu, logvar)

                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(p.data, alpha=0.001)

                if gen_iterations % 100 == 0:
                    # print(D_logs + '\n' + G_logs)
                    logger.info(D_logs + '\n' + G_logs)

                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)

            end_t = time.time()

            # print('''[%d/%d][%d]
            #       Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
            #       % (epoch, self.max_epoch, self.num_batches,
            #          errD_total, errG_total,
            #          end_t - start_t))
            logger.info('''[%d/%d][%d]
                        Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                        % (epoch, self.max_epoch, self.num_batches,
                        errD_total, errG_total,
                        end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0: 
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

        display_loss("../output/log.txt",self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir, split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            # print('Load text encoder from:', cfg.TRAIN.NET_E)
            logger.info('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # load image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            # print('Load G from: ', model_dir)
            logger.info('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            R = np.zeros(30000)
            cont = True
            idx = 0 ###
            for ii in range(1):  # 重复5次，每次是生成全部
                if (cont == False):
                    break
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if (cont == False):
                        break
                    if step % 100 == 0:
                        logger.info('step: %s', step)

                    # for i in range(0, len(data[2])):  # 把 5 句话分开放，每个batach=8
                    #     temp = [data[0], data[1][:,i,:,:], data[2][i], data[3], data[4],
                    #             data[5][:,i,:,:], data[6][i], data[7]]

                    imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)

                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        im = fake_imgs[k][j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                        idx = idx+1
                        im.save(fullpath)
                        im.save(fullpath)

                    _, cnn_code = image_encoder(fake_imgs[-1])
                    for i in range(batch_size):
                        mis_captions, mis_captions_len = self.dataset.get_mis_caption(class_ids[i])
                        hidden = text_encoder.init_hidden(99)
                        _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                        rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                        ### cnn_code = 1 * nef
                        ### rnn_code = 100 * nef
                        scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                        cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                        rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                        norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                        scores0 = scores / norm.clamp(min=1e-8)
                        if torch.argmax(scores0) == 0:
                            R[R_count] = 1
                        R_count += 1

                    ####### R-precision #########
                    if R_count >= 30000:
                        sum = np.zeros(10)
                        np.random.shuffle(R)
                        for i in range(10):
                            sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                        R_mean = np.average(sum)
                        R_std = np.std(sum)
                        print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std)) ## output r-precision 
                        cont = False

    def gen_description(self, data_dic, juzi):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)

            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            imgs_per_sent = 16
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                print(save_dir)

                for i in range(len(juzi)):
                    captions, cap_lens, sorted_indices = data_dic[key]

                    batch_size = captions.shape[0]
                    nz = cfg.GAN.Z_DIM
                    aa = np.array([captions[i]])
                    bb = np.array([cap_lens[0]])
                    captions = Variable(torch.from_numpy(aa), volatile=True)
                    cap_lens = Variable(torch.from_numpy(bb), volatile=True)

                    captions = captions.cuda()
                    cap_lens = cap_lens.cuda()


                    hidden = text_encoder.init_hidden(1)
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)

                    # words_embs_xx = torch.cat([clip.tokenize(juzi[0])]).cuda()
                    text_inputs = torch.cat([clip.tokenize(juzi[sorted_indices[i]])]).cuda()
                    text_features = self.clip_loss.model.encode_text(text_inputs)
                    # text_features /= text_features.norm(dim=-1, keepdim=True)
                    # text_embeds = text_features.expand(8, -1)

                    x_recons = []
                    all_fake_imgs = []
                    all_attention_maps = []
                    # all_att_c_maps = []
                    for t in range(imgs_per_sent):   # 每个句子生成5张图像，选择clip相似度最高的图像进行保存
                        noise = Variable(torch.FloatTensor(1, nz), volatile=True)
                        noise = noise.cuda()
                        noise.data.normal_(0, 1)

                        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                        x_recons.append(fake_imgs[-1])
                        all_fake_imgs.append(fake_imgs)
                        all_attention_maps.append(attention_maps)
                        # all_att_c_maps.append(attention_c_maps)
                    # torch.cuda.empty_cache()
                    x_recon = torch.cat(x_recons, dim=0)

                    ##################################
                    # filter by CLIP
                    ##################################
                    img_embeddings = []
                    for t in range(imgs_per_sent):
                        tensor_img_to_PIL = ToPILImage()(x_recon[t][0])
                        image_input = self.clip_loss.preprocess(tensor_img_to_PIL).unsqueeze(0).to("cuda")
                        embd = self.clip_loss.model.encode_image(image_input)  # [B, 512]
                        img_embeddings.append(embd)
                        torch.cuda.empty_cache()
                    img_embeddings = torch.cat(img_embeddings, dim=0)

                    sim = F.cosine_similarity(text_features, img_embeddings)
                    topk = sim.argsort(descending=True)  # 排序后的索引号
                    sim_sort = sorted(sim.cpu().data.numpy(), reverse=True)
                    print("CLIP similarity", sim_sort)

                    cap_lens_np = cap_lens.cpu().data.numpy()

                    # fake_imgs = all_fake_imgs[topk]  # 3个，分辨率由低到高
                    # attention_maps = all_attention_maps[topk]
                    for j in range(imgs_per_sent):  # 一条句子一条句子的去生成和处理，从最长的句子开始
                        save_name = '%s/%d_s_%d' % (save_dir, sorted_indices[i], j)
                        for k in range(3):
                            im = all_fake_imgs[topk[j]][k][0].data.cpu().numpy() # 对于每个句子生成的5张图像，按照给语义相似度由高到低进行输出保存
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            im = np.transpose(im, (1, 2, 0))
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(all_attention_maps[topk[j]])):
                            if len(all_fake_imgs[topk[j]]) > 1:
                                im = all_fake_imgs[topk[j]][k + 1][0].detach().cpu()
                            else:
                                im = all_fake_imgs[topk[j]][0][0].detach().cpu()
                            attn_maps = all_attention_maps[topk[j]][k][0]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = build_super_images2(im.unsqueeze(0),
                                                                     captions[0].unsqueeze(0),
                                                                     [cap_lens_np[0]], self.ixtoword,
                                                                     [attn_maps], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)

                        # for k in range(len(all_att_c_maps[topk[j]])):
                        #     if len(all_fake_imgs[topk[j]]) > 1:
                        #         im = all_fake_imgs[topk[j]][k + 1][0].detach().cpu()
                        #     else:
                        #         im = all_fake_imgs[topk[j]][0][0].detach().cpu()
                        #     attn_maps = all_att_c_maps[topk[j]][k][0]
                        #     att_sze = attn_maps.size(1)
                        #     img_set, sentences = build_super_images2(im.unsqueeze(0),
                        #                                              captions[0].unsqueeze(0),
                        #                                              [cap_lens_np[0]], self.ixtoword,
                        #                                              [attn_maps], att_sze)
                        #     if img_set is not None:
                        #         im = Image.fromarray(img_set)
                        #         fullpath = '%s_c_a%d.png' % (save_name, k)
                        #         im.save(fullpath)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()

                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)

                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            im = np.transpose(im, (1, 2, 0))
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = build_super_images2(im[j].unsqueeze(0),
                                                                     captions[j].unsqueeze(0),
                                                                     [cap_lens_np[j]], self.ixtoword,
                                                                     [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
