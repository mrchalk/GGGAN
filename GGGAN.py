import os
import cv2
import h5py
import functools
import numpy as np
#import pydicom as dcm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.autograd import Variable

#  define norm layer and activation layer

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(LayerNorm, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not implemented' % norm_type)
    return norm_layer

def get_activation(act_type='relu'):
    if act_type == 'lrelu':
        activation  = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif act_type == 'relu':
        activation  = nn.ReLU(inplace=True)
    else:
        raise NotImplementedError('activation layer [%s] is not implemented' % act_type)
    return activation

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.weight.requires_grad:
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# define generator

class GroupResBlock(nn.Module):
    def __init__(self, ndf, norm_layer, activation):
        super(GroupResBlock, self).__init__()
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(ndf, ndf, kernel_size=3, padding=0), norm_layer(ndf), activation,
                  nn.ReflectionPad2d(1), nn.Conv2d(ndf, ndf, kernel_size=3, padding=0), norm_layer(ndf)]
        self.block = nn.Sequential(*layers)
        self.act = activation

    def forward(self, x):
        out = x + self.block(x)
        return self.act(out)

class GroupConv(nn.Module):
    def __init__(self, input_factor, output_factor):
        super(GroupConv, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(input_factor, output_factor, kernel_size=3, padding=0)
        self.input_factor = input_factor

    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(-1, self.input_factor, w, h)
        x = self.pad(x)
        x = self.conv(x)
        x = x.view(n, -1, w, h)
        return x

class GroupRefineBlock(nn.Module):
    def __init__(self, ndf, factor, pre_factor, norm_layer, activation, resblock_num=0, input_nc=96):
        super(GroupRefineBlock, self).__init__()
        layers = []
        for i in range(resblock_num):
            layers += [GroupResBlock(ndf*factor, norm_layer, activation)]

        if factor != 2:
            layers += [GroupConv(factor, pre_factor*4), nn.PixelShuffle(2), activation]
            self.embedding = nn.Sequential(*[GroupConv(input_nc/ndf, pre_factor), norm_layer(ndf*pre_factor), activation])
            self.merge = nn.Sequential(*[GroupConv(pre_factor*2, pre_factor), norm_layer(ndf*pre_factor), activation])
        else:
            layers += [nn.ReflectionPad2d(1), nn.Conv2d(ndf*2, ndf*2, kernel_size=3, padding=0), activation,
                       nn.ReflectionPad2d(3), nn.Conv2d(ndf*2, 1, kernel_size=7, padding=0), nn.Tanh()]

        self.block = nn.Sequential(*layers)
        self.is_output_block = factor == 2

    def forward(self, x, skip):
        x = self.block(x)
        if not self.is_output_block:
            skip = self.embedding(skip)
            cat = Variable(torch.FloatTensor(x.size(0), 2*x.size(1), x.size(2), x.size(3)))
            if x.is_cuda:
                cat = cat.cuda()
            cat[:, ::2, :, :] = skip
            cat[:, 1::2, :, :] = x
            cat = self.merge(cat)
            return cat
        return x

def GaussianKernel(kernel_size=7, sigma=1.6):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2
    s = 2*(sigma**2)
    sum_val = 0.
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / s)
            sum_val += kernel[i, j]
    kernel /= sum_val
    return kernel

class DownsampleBlock(nn.Module):
    def __init__(self):
        super(DownsampleBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=0, bias=False)
        self.conv.weight.data = torch.FloatTensor(GaussianKernel()).view(1, 1, 7, 7)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(-1, 1, w, h)
        x = self.pad(x)
        x = self.conv(x)
        x = self.down(x)
        x = x.view(n, c, w/2, h/2)
        return x

class GroupCascadedRefine(nn.Module):
    def __init__(self, input_nc, ndf, stages, norm_layer, activation, resblock_num=0):
        super(GroupCascadedRefine, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)#DownsampleBlock()
        blocks = []
        factor = 1
        for i in range(stages):
            pre_factor = factor
            factor = min(32, 2*factor)
            blocks.insert(0, GroupRefineBlock(ndf, factor, pre_factor, norm_layer, activation, resblock_num, input_nc))
        self.group_refine_blocks = nn.ModuleList(blocks)
        self.embedding = nn.Sequential(*[GroupConv(input_nc/ndf, factor), norm_layer(ndf*factor), activation])

    def forward(self, x):
        multiscale = [None]
        for i in range(len(self.group_refine_blocks)-1):
            multiscale.insert(0, x)
            x = self.down(x)
        x = self.embedding(x)
        for i in range(len(self.group_refine_blocks)):
            x = self.group_refine_blocks[i](x, multiscale[i])
        return x

def define_groupG(pretrain_path=None, use_gpu=True):
    norm_layer = get_norm_layer('layer')
    activation = get_activation('lrelu')
    netG = GroupCascadedRefine(96, 32, 6, norm_layer, activation, 3)
    if use_gpu:
        netG = torch.nn.DataParallel(netG)
        netG = netG.cuda()
    if pretrain_path is not None:
        pretrained_dict = torch.load(pretrain_path)
        net_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        print len(pretrained_dict)
        netG.load_state_dict(pretrained_dict)
    else:
        netG.apply(weights_init)
    return netG

# define discriminator

class GradConv(nn.Module):
    def __init__(self, input_nc, dilation):
        super(GradConv, self).__init__()
        self.pad = nn.ReflectionPad2d(dilation)
        self.conv = nn.Conv2d(input_nc, 4, kernel_size=3, padding=0, dilation=dilation)
        self.conv.weight.data = torch.FloatTensor([[[[-1,-2,-1], [0,0,0], [1,2,1]]], [[[-1,0,1], [-2,0,2], [-1,0,1]]],
                                                   [[[-2,-1,0], [-1,0,1], [0,1,2]]], [[[0,-1,-2], [1,0,-1], [2,1,0]]]])
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class GradBlock(nn.Module):
    def __init__(self, input_nc=1):
        super(GradBlock, self).__init__()
        self.block = nn.ModuleList([GradConv(input_nc, 1), GradConv(input_nc, 3), GradConv(input_nc, 5)])
        for param in self.block.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = torch.cat((x, self.block[0](x)), 1)
        out = torch.cat((out, self.block[1](x)), 1)
        out = torch.cat((out, self.block[2](x)), 1)
        return out

class MergeBlock(nn.Module):
    def __init__(self, input_nc, ndf, activation):
        super(MergeBlock, self).__init__()
        self.conv1 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(input_nc[0], ndf/2, kernel_size=4, stride=2, padding=0)])
        self.conv2 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(input_nc[1], ndf/2, kernel_size=4, stride=2, padding=0)])
        self.act = activation

    def forward(self, input):
        out = torch.cat((self.conv1(input[0]), self.conv2(input[1])), 1)
        return self.act(out)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, norm_layer, activation, ndf=64, n_layers=3, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        sequence = [[MergeBlock(input_nc, ndf, activation)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.ReflectionPad2d(1), nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=0),
                          norm_layer(nf), activation]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.ReflectionPad2d(1), nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=0), norm_layer(nf), activation]]

        sequence += [[nn.ReflectionPad2d(1), nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding=0)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, norm_layer, activation, ndf=64, n_layers=3, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, norm_layer, activation, ndf, n_layers, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled[0] = self.downsample(input_downsampled[0])
                input_downsampled[1] = self.downsample(input_downsampled[1])
        return result

def define_D(input_nc, norm_type='instance', act_type='lrelu', ndf=64, n_layers=3, use_sigmoid=False,
             num_D=3, getIntermFeat=False, use_gpu=True, pretrain_path=None):
    norm_layer = get_norm_layer(norm_type)
    activation = get_activation(act_type)
    netD = MultiscaleDiscriminator(input_nc, norm_layer, activation, ndf=ndf, n_layers=n_layers,
                                   use_sigmoid=use_sigmoid, num_D=num_D, getIntermFeat=getIntermFeat)
    if use_gpu:
        netD = torch.nn.DataParallel(netD)
        netD = netD.cuda()
    if pretrain_path is not None:
        pretrained_dict = torch.load(pretrain_path)
        net_dict = netD.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        print len(pretrained_dict)
        netD.load_state_dict(pretrained_dict)
    else:
        netD.apply(weights_init)
    return netD

# define GAN loss and feature loss

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, use_gpu=True):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.use_gpu = use_gpu
        if use_lsgan:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            if self.real_label_var is None or self.real_label_var.numel() != input.numel():
                real_tensor = torch.FloatTensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor)
            target_tensor = self.real_label_var
        else:
            if self.fake_label_var is None or self.fake_label_var.numel() != input.numel():
                fake_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor)
            target_tensor = self.fake_label_var
        if self.use_gpu:
            target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0.
            scale_lambda = 1.
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += scale_lambda * self.criterion(pred, target_tensor)
                scale_lambda *= 0.5
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.criterion(input[-1], target_tensor)

class FeatLoss(nn.Module):
    def __init__(self, n_layers=3, num_D=3, lambda_feat=10.0):
        super(FeatLoss, self).__init__()
        self.feat_weights = 4.0 / (n_layers + 1)
        self.D_weights = 1.0 / num_D
        self.num_D = num_D
        self.lambda_feat = lambda_feat
        self.criterionFeat = nn.L1Loss()
        #self.criterionFeat = nn.MSELoss()

    def __call__(self, pred_fake, pred_real):
        loss = 0.
        scale_lambda = 1.
        for i in range(self.num_D):
            for j in range(len(pred_fake[i])-1):
                loss += scale_lambda * self.D_weights * self.feat_weights  * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.lambda_feat
            scale_lambda *= 0.5
        return loss

# train GGGAN

def train_GGGAN(batch_size, dataset_path, epoch_num, lr, DBT_channels=96, DM_channels=1, start_epoch=0):
    train_dir = 'GGGAN/'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    weights_dir = train_dir + 'weights/'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    file_list = os.listdir(dataset_path)
    dataset_size = len(file_list)
    dataloader_D = define_dataloader(batch_size, dataset_path, file_list, DBT_channels=DBT_channels)
    dataloader_G = define_dataloader(batch_size, dataset_path, file_list, DBT_channels=DBT_channels)

    cudnn.benchmark = True
    pretrain_path_G = None
    pretrain_path_D = None
    iters = 0
    losses = np.zeros((4, epoch_num * dataset_size / 1000), dtype=np.float32)
    if start_epoch != 0:
        pretrain_path_G = weights_dir + 'GGGAN_%d_netG.pth'%(start_epoch-1)
        pretrain_path_D = weights_dir + 'GGGAN_%d_netD.pth'%(start_epoch-1)
        iters = start_epoch * dataset_size
        losses[:, :iters/1000] = np.load(train_dir+'losses.npy')[:, :iters/1000]
    netG = define_groupG(pretrain_path=pretrain_path_G)
    netD = define_D([DBT_channels, DM_channels], getIntermFeat=True, pretrain_path=pretrain_path_D)
    criterionGAN = GANLoss(use_lsgan=True)
    criterionFeat = FeatLoss(n_layers=3, num_D=3)
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=lr, betas=(0.5, 0.9))
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=lr, betas=(0.5, 0.9))

    print 'Training'
    netG.train()
    netD.train()
    for epoch in range(start_epoch, epoch_num):
        losses_D_fake = []
        losses_D_real = []
        losses_G_GAN = []
        losses_G_Feat = []
        iterator = iter(dataloader_G)
        for batch, (dbt, dm) in enumerate(dataloader_D):
            dbt = Variable(dbt.cuda())
            dm = Variable(dm.cuda())
            for p in netG.parameters():
                p.requires_grad = False
            netG.eval()
            optimizerD.zero_grad()
            fake_image = netG(dbt)
            pred_fake_pool = netD([dbt, fake_image.detach()])
            loss_D_fake = criterionGAN(pred_fake_pool, False)
            pred_real = netD([dbt, dm])
            loss_D_real = criterionGAN(pred_real, True)
            losses_D_fake.append(loss_D_fake.data[0])
            losses_D_real.append(loss_D_real.data[0])
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizerD.step()
            for p in netG.parameters():
                p.requires_grad = True
            netG.train()

            dbt, dm = next(iterator)
            dbt = Variable(dbt.cuda())
            dm = Variable(dm.cuda())
            for p in netD.parameters():
                p.requires_grad = False
            netD.eval()
            optimizerG.zero_grad()
            fake_image = netG(dbt)
            pred_fake = netD([dbt, fake_image])
            pred_real = netD([dbt, dm])
            loss_G_GAN = criterionGAN(pred_fake, True)
            loss_G_Feat = criterionFeat(pred_fake, pred_real)
            losses_G_GAN.append(loss_G_GAN.data[0])
            losses_G_Feat.append(loss_G_Feat.data[0])
            loss_G = loss_G_GAN + loss_G_Feat
            loss_G.backward()
            optimizerG.step()
            for p in netD.parameters():
                p.requires_grad = True
            netD.train()

            iters += batch_size
            if iters % 1000 == 0:
                losses_D_fake = sum(losses_D_fake) / len(losses_D_fake)
                losses_D_real = sum(losses_D_real) / len(losses_D_real)
                losses_G_GAN = sum(losses_G_GAN) / len(losses_G_GAN)
                losses_G_Feat = sum(losses_G_Feat) / len(losses_G_Feat)
                print 'Iters: %d D fake: %.4f D Real: %.4f G GAN: %.4f G Feat: %.4f'%(iters,
                       losses_D_fake, losses_D_real, losses_G_GAN, losses_G_Feat)
                losses[0, iters/1000-1] = losses_D_fake
                losses[1, iters/1000-1] = losses_D_real
                losses[2, iters/1000-1] = losses_G_GAN
                losses[3, iters/1000-1] = losses_G_Feat
                pandas.DataFrame({'D fake': losses[0, :iters/1000], 'D Real': losses[1, :iters/1000], 'G GAN': losses[2, :iters/1000],
                                  'G Feat': losses[3, :iters/1000]}).plot().get_figure().savefig(train_dir+'losses.png')
                losses_D_fake = []
                losses_D_real = []
                losses_G_GAN = []
                losses_G_Feat = []

        torch.save(netG.state_dict(), weights_dir + 'GGGAN_%d_netG.pth'%epoch)
        torch.save(netD.state_dict(), weights_dir + 'GGGAN_%d_netD.pth'%epoch)
        np.save(train_dir+'losses.npy', losses[:, :iters/1000])
