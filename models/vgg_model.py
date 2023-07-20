import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import os
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import normalize as l2_normalize
from util.image_pool import ImagePool
import torch.nn as nn
import torchvision.models as models


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        #for param in self.parameters():
        #    param.requires_grad = False

    def forward(self, x, layers=None, encode_only=False, resize=False):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        if encode_only:
            if len(layers) > 0:
                feats = []
                for layer, key in enumerate(out):
                    if layer in layers:
                        feats.append(out[key])
                return feats
            else:
                return out['relu3_1']
        return


class VGGModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_idt', type=float, default=5.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_path', type=float, default=0.01, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--path_layers', type=str, default='0,3,6,10,14', help='compute NCE loss on which layers')
        parser.add_argument('--style_dim', type=int, default=8, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--path_interval_min', type=float, default=0.05, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--path_interval_max', type=float, default=0.10, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--noise_std', type=float, default=1.0, help='compute NCE loss on which layers')
        parser.add_argument('--tag', type=str, default='debug', help='compute NCE loss on which layers')
        parser.set_defaults(no_html=True, pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        model_id = '%s' % opt.tag
        model_id += '/'+os.path.basename(opt.dataroot.strip('/')) + '_%s' % opt.direction
        model_id += '/lam%s_layers%s_dim%d_rec%d_idt%s_pool%d_noise%s_kl%s' % \
            (opt.lambda_path, opt.path_layers, opt.style_dim, opt.lambda_rec, opt.lambda_idt, opt.pool_size, opt.noise_std, opt.lambda_kl)
        #model_id += '_normalize' if opt.normalize else '_vanilla'
        #model_id += '_auto' if opt.auto_interval else '_fix'

        parser.set_defaults(name=model_id)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G_rec', 'G_idt', 'G_kl', 'G_path',  'd1', 'd2']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.path_layers = [int(i) for i in self.opt.path_layers.split(',')]
        for l in self.path_layers:
            self.loss_names += ['energy_%d' % l]

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.d_A = torch.zeros([1]).to(self.device)
        self.d_B = torch.ones([1]).to(self.device)
        print(self.netG)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            print(self.netD)
            self.netPre = VGG16().to(self.device)
            self.normalization = Normalization(self.device)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real = torch.cat([self.real_A, self.real_B], dim=0)
        latents = self.netG(real, mode='encode')
        if self.isTrain and self.opt.noise_std > 0:
            noise = torch.ones_like(latents).normal_(mean=0, std=self.opt.noise_std)
            self.mu = latents
            latents = latents + noise
        self.latent_A, self.latent_B = latents.chunk(2, dim=0)
        ds = torch.cat([self.d_A, self.d_B, self.d_B], 0).unsqueeze(-1)
        latents = torch.cat([self.latent_A, self.latent_A, self.latent_B], 0)
        images = self.netG((latents, ds), mode='decode')
        self.rec_A, self.fake_B, self.idt_B = images.chunk(3, dim=0)


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B_pool.query(self.fake_B.detach())
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True).mean()
        self.loss_G_rec = self.criterionIdt(self.rec_A, self.real_A).mean()
        self.loss_G_idt = self.criterionIdt(self.idt_B, self.real_B).mean()
        self.loss_d1 = torch.tensor(0.)
        self.loss_d2 = torch.tensor(0.)
        for l in self.path_layers:
            setattr(self, 'loss_energy_%d' % l, 0)
        if self.opt.noise_std>0:
            self.loss_G_kl = torch.pow(self.mu, 2).mean()
        else:
            self.loss_G_kl = 0
        self.loss_n_dots = 0
        if self.opt.lambda_path > 0:
            self.loss_G_path = self.compute_path_losses()
        else:
            self.loss_G_path = torch.tensor(0.)
        self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN +\
                      self.opt.lambda_rec * self.loss_G_rec + \
                      self.opt.lambda_idt * self.loss_G_idt + \
                      self.opt.lambda_kl * self.loss_G_kl + \
                      self.opt.lambda_path * self.loss_G_path
        return self.loss_G

    def compute_path_losses(self):
        norm_rec_A = self.normalization((self.rec_A+1)*0.5)
        norm_fake_B = self.normalization((self.fake_B+1)*0.5)
        feats_src = self.netPre(norm_rec_A, [4,7,9], encode_only=True)
        feats_tgt = self.netPre(norm_fake_B, [4,7,9], encode_only=True)
        total_loss = 0.0
        for i, (feat_src, feat_tgt) in enumerate(zip(feats_src, feats_tgt)):
            loss = (feat_src-feat_tgt)**2
            total_loss += loss.mean()
        return total_loss

    @torch.no_grad()
    def interpolation(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'AtoB':
            x = x_a
        else:
            x = x_b
        interps = []
        for i in range(min(x.size(0),8)):
            h_a = self.netG(x[i].unsqueeze(0), mode='encode')
            d = 0.0
            local_interps = []
            local_interps.append(x[i].unsqueeze(0))
            while d <= 1.:
                d_t = torch.tensor([d]).to(x_a.device).unsqueeze(-1)
                local_interps.append(self.netG((h_a, d_t), mode='decode'))
                d += 0.1
            local_interps = torch.cat(local_interps, 0)
            interps.append(local_interps)
        self.netG.train()
        return interps

    @torch.no_grad()
    def translate(self, x):
        self.netG.eval()
        h = self.netG(x, mode='encode')
        out = self.netG((h, self.d_B), mode='decode')
        self.netG.train()
        return out

    @torch.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'BtoA':
            x_a, x_b = x_b, x_a
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a = self.netG(x_a[i].unsqueeze(0), mode='encode')
            h_b = self.netG(x_b[i].unsqueeze(0), mode='encode')
            x_a_recon.append(self.netG((h_a, self.d_A), mode='decode'))
            x_b_recon.append(self.netG((h_b, self.d_B), mode='decode'))
            x_ab.append(self.netG((h_a, self.d_B), mode='decode'))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ab = torch.cat(x_ab)
        self.netG.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon

