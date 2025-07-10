import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from .reg import Transformer_2D, smooothing_loss, Reg

def laplace_nll(real_B, fake_B, sigma_min=1e-3):
    C = torch.log(torch.tensor(2.0))
    n = real_B.shape[1]
    
    mu = fake_B[:, :n, :, :]
    sigma = fake_B[:, n:, :, :]

    # Ensure sigma is positive and above a minimum threshold
    sigma = torch.clamp(sigma, min=sigma_min)
    
    # Compute the negative log-likelihood
    nll = torch.abs((mu - real_B) / sigma) + torch.log(sigma) + C
    nll_mean = torch.mean(nll)
    
    return nll_mean

def gradient_mae_loss(pred, target, eps=1e-8):
    """
    Compute Mean Absolute Error between gradients of predicted and target images.
    
    Args:
        pred: Predicted image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W]
        eps: Small epsilon to avoid numerical issues
    
    Returns:
        Gradient MAE loss
    """
    # Compute gradients using Sobel operators
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    device = pred.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)
    
    # Expand sobel kernels to match number of channels
    channels = pred.shape[1]
    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)
    
    # Compute gradients for predicted image
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
    pred_gradient_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + eps)
    
    # Compute gradients for target image
    target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
    target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
    target_gradient_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2 + eps)
    
    # Check for NaN values and handle them
    if torch.isnan(pred_gradient_magnitude).any() or torch.isnan(target_gradient_magnitude).any():
        print("Warning: NaN detected in gradient computation")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute MAE between gradient magnitudes
    gradient_mae = torch.mean(torch.abs(pred_gradient_magnitude - target_gradient_magnitude))
    
    return gradient_mae

class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data."""
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # Store options
        self.netG_reps = opt.netG_reps
        self.use_reg = opt.use_reg
        self.use_nll = opt.use_nll
        self.use_trans = opt.use_trans
        self.grad_clip = opt.grad_clip
        self.sigma_min = opt.sigma_min
        self.lambda_grad = opt.lambda_grad

        # Only initialize registration components if registration is enabled
        if self.use_reg:
            # init the transform by the registration network
            print(f"Initializing registration network (use_reg={self.use_reg})")

            if self.use_nll:
                self.R_A = Reg(int(opt.netG.split('_')[-1]), int(opt.netG.split('_')[-1]), opt.output_nc//2, opt.output_nc//2).cuda()
            else:
                self.R_A = Reg(int(opt.netG.split('_')[-1]), int(opt.netG.split('_')[-1]), opt.output_nc, opt.output_nc).cuda()

            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # specify the training losses you want to print out
        #self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the training losses you want to print out
        self.loss_names = ['G_GAN', 'G_L1', 'G_Grad', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, netG_reps=self.netG_reps, use_trans=self.use_trans)
        # get netD_weight
        self.netD_mult = opt.netD_mult

        if self.isTrain:  
            # define a discriminator
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_nll=self.use_nll)


            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionNLL = laplace_nll
            self.criterionGrad = gradient_mae_loss

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass."""
        self.fake_B, *_ = self.netG(self.real_A)  # G(A)
        # clamp the input to the range [0, 1]
        self.fake_B = torch.clamp(self.fake_B, 0, 1)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        n = self.real_B.shape[1]
        #if self.use_nll:
        #    self.mu = self.fake_B[:, :n, :, :]
        #else:
        #    self.mu = self.fake_B
        self.mu = self.fake_B[:, :n, :, :]

        fake_AB = torch.cat((self.real_A, self.mu), 1)  # Use only mean for D
        
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        n = self.real_B.shape[1]

        self.mu = self.fake_B[:, :n, :, :]

        fake_AB = torch.cat((self.real_A, self.mu), 1)  # Use only mean for D

        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        if self.use_reg:
            # Registration pathway
            Trans = self.R_A(self.mu, self.real_B)  # get transformation matrix
            SysRegist_A2B = self.spatial_transform(self.fake_B, Trans)
            self.mu_reg = SysRegist_A2B[:, :n, :, :]
            
            if self.use_nll:
                self.loss_G_NLL = self.criterionNLL(self.real_B, SysRegist_A2B, self.sigma_min)  # NLL loss with reg
            self.loss_G_L1 = self.criterionL1(self.mu_reg, self.real_B) * self.opt.lambda_L1  # L1 loss with reg
            
            # Add gradient loss with registration
            self.loss_G_Grad = self.criterionGrad(self.mu_reg, self.real_B) * self.lambda_grad
            
            # Smooth loss
            Smooth_lamda = 10
            SM_loss = Smooth_lamda * smooothing_loss(Trans)
            
            # Combined loss with registration
            if self.use_nll:
                self.loss_G = self.loss_G_NLL + self.netD_mult * self.loss_G_GAN + SM_loss + self.loss_G_Grad
            else:
                self.loss_G = self.loss_G_L1 + self.netD_mult * self.loss_G_GAN + SM_loss + self.loss_G_Grad
                
        else:
            # Standard pathway (like in pix2pix_model.py)
            if self.use_nll:
                self.loss_G_NLL = self.criterionNLL(self.real_B, self.fake_B, self.sigma_min)
            self.loss_G_L1 = self.criterionL1(self.mu, self.real_B) * self.opt.lambda_L1
            
            # Add gradient loss without registration
            self.loss_G_Grad = self.criterionGrad(self.mu, self.real_B) * self.lambda_grad
            
            # Standard combined loss
            if self.use_nll:
                self.loss_G = self.loss_G_NLL + self.netD_mult * self.loss_G_GAN + self.loss_G_Grad
            else:
                self.loss_G = self.loss_G_L1 + self.netD_mult * self.loss_G_GAN + self.loss_G_Grad
            
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()     # set G's gradients to zero
        
        if self.use_reg:
            self.optimizer_R_A.zero_grad()  # set registration's gradients to zero
            
        self.backward_G()  # calculate gradients for G
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.grad_clip)   # clip gradients to avoid exploding gradients
        self.optimizer_G.step()  # update G's weights
        
        if self.use_reg:
            self.optimizer_R_A.step()  # update registration's weights



