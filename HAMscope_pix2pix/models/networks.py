import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from einops import rearrange


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()

    elif norm_type == 'layer':
        # Make LayerNorm2d work like the successful TransformerBlock LayerNorm
        class LayerNorm2d(nn.Module):
            def __init__(self, num_features):
                super().__init__()
                # Use the same approach as TransformerBlock
                self.layer_norm = nn.LayerNorm(num_features)

            def forward(self, x):
                # x shape: [B, C, H, W]
                B, C, H, W = x.shape
                # Reshape to [B, H*W, C] like in TransformerBlock
                x_reshaped = x.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
                # Apply LayerNorm over channel dimension only (like TransformerBlock)
                x_normalized = self.layer_norm(x_reshaped)  # Normalize over C dimension
                # Reshape back to [B, C, H, W]
                return x_normalized.permute(0, 2, 1).view(B, C, H, W)

        norm_layer = LayerNorm2d
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_shape=(64, 64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_shape = max_shape
        
        # Pre-compute positional encodings
        max_h, max_w = max_shape
        pe = torch.zeros(max_h * max_w, embed_dim)
        
        position = torch.arange(0, max_h * max_w).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, shape, device):
        h, w = shape
        if h * w > self.pe.size(0):
            # Fallback for larger shapes
            return torch.zeros(h * w, self.embed_dim, device=device)
        return self.pe[:h * w].to(device)


class AttnFF(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.l1 = nn.Linear(self.in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_channels, self.out_channels)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class TransformerBlock(nn.Module):
    """
    A self-attention transformer block optimized for smaller feature maps in U-Net bottleneck.
    """
    def __init__(self, in_channels, num_heads=8, max_shape=(64, 64)):
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})")

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_channels_ff = 2 * in_channels  # Reduced from 4x to 2x
        self.pos_encoder = PositionalEncoding(in_channels, max_shape)

        self.norm_1 = nn.LayerNorm(in_channels)
        self.norm_2 = nn.LayerNorm(in_channels)

        # Use more efficient attention for smaller sequences
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=False,
            dropout=0.1  # Add some dropout for regularization
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, self.hidden_channels_ff),
            nn.GELU(),  # More efficient than ReLU for transformers
            nn.Dropout(0.1),
            nn.Linear(self.hidden_channels_ff, in_channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Only warn for very large feature maps (shouldn't happen with depth restriction)
        if H * W > 4096:  # 64x64 threshold
            print(f"Warning: Feature map {H}x{W} larger than expected for bottom-half transformer")

        # Reshape for attention: (B, C, H, W) -> (H*W, B, C)
        x_seq = x.view(B, C, H * W).permute(2, 0, 1)
        x_residual = x_seq

        # Pre-norm + positional encoding + attention
        x_seq = self.norm_1(x_seq)
        pos_embedding = self.pos_encoder((H, W), x.device)
        x_seq = x_seq + pos_embedding.unsqueeze(1)

        attn_output, _ = self.attention(query=x_seq, key=x_seq, value=x_seq)
        x_seq = x_residual + attn_output

        # Pre-norm + feed forward
        x_residual_ff = x_seq
        x_seq = self.norm_2(x_seq)
        ff_output = self.feed_forward(x_seq)
        x_seq = ff_output + x_residual_ff

        return x_seq.permute(1, 2, 0).view(B, C, H, W)


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], netG_reps=1, use_trans=0, transformer_depth_threshold=4):
    """Create a generator"""

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, netG_reps=netG_reps, use_trans=use_trans, transformer_depth_threshold=transformer_depth_threshold)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, netG_reps=netG_reps, use_trans=use_trans, transformer_depth_threshold=transformer_depth_threshold)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, netG_reps=netG_reps, use_trans=use_trans, transformer_depth_threshold=transformer_depth_threshold)
    elif netG == 'unet_1024':
        net = UnetGenerator(input_nc, output_nc, 10, ngf, norm_layer=norm_layer, use_dropout=use_dropout, netG_reps=netG_reps, use_trans=use_trans, transformer_depth_threshold=transformer_depth_threshold)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
    # After creating the generator
    print_model_parameters(net, "UNet Generator")

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], use_nll = False):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_nll=use_nll)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_nll=use_nll)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, netG_reps=1, use_trans=0, transformer_depth_threshold=4):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_attention (bool) -- whether to use attention modules
            use_trans (int) -- type of transformer (1=Spectral, 2=Transformer)
            transformer_depth_threshold (int) -- minimum depth for transformer application
        """
        super(UnetGenerator, self).__init__()
        self.netG_reps = netG_reps
        self.transformer_depth_threshold = transformer_depth_threshold
        self.input_nc = input_nc
        self.output_nc = output_nc
        
        print(f"UnetGenerator: transformer_depth_threshold={transformer_depth_threshold}")
        print(f"UnetGenerator: applying transformers to depths >= {transformer_depth_threshold}")
        print(f"UnetGenerator: netG_reps={netG_reps}, input_nc={input_nc}, output_nc={output_nc}")

        # Use output_nc as input_nc if we have multi-repetition (since output becomes input)
        effective_input_nc = output_nc if netG_reps > 1 else input_nc
        
        # Add input adaptation layer if needed
        if netG_reps > 1 and input_nc != output_nc:
            self.input_adapter = nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0, bias=False)
            print(f"Added input adapter: {input_nc} -> {output_nc} channels")
        else:
            self.input_adapter = None

        
        # construct unet structure with depth tracking
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_trans=use_trans, depth=num_downs-1, transformer_depth_threshold=transformer_depth_threshold)
        
        for i in range(num_downs - 5):
            current_depth = num_downs - 2 - i
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_trans=use_trans, depth=current_depth, transformer_depth_threshold=transformer_depth_threshold)
        
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_trans=use_trans, depth=4, transformer_depth_threshold=transformer_depth_threshold)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_trans=use_trans, depth=3, transformer_depth_threshold=transformer_depth_threshold)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_trans=use_trans, depth=2, transformer_depth_threshold=transformer_depth_threshold)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=effective_input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_trans=use_trans, depth=1, transformer_depth_threshold=transformer_depth_threshold)

    def forward(self, x, feats=None):
        # Adapt input channels for first pass if needed
        if self.input_adapter is not None and feats is None:
            x = self.input_adapter(x)
        
        out, collected_feats = self.model(x, feats=feats)
        for _ in range(self.netG_reps - 1):
            out, _ = self.model(out, feats=collected_feats)
        return out, collected_feats


# pass feature maps
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection, modified to return intermediate features and accept them."""

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_trans=0, depth=0, transformer_depth_threshold=4):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.submodule = submodule
        self.depth = depth
        self.transformer_depth_threshold = transformer_depth_threshold

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, inplace=False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=False)
        upnorm = norm_layer(outer_nc)

        # Apply attention based on configurable depth threshold
        apply_attention = not outermost and use_trans > 0 and depth >= self.transformer_depth_threshold
        
        if apply_attention:
            if use_trans == 1:
                self.attention = Spectral_Atten(dim=input_nc, heads=4)
                print(f"✓ Adding SPECTRAL attention to block: depth={depth}")
            elif use_trans == 2:
                num_heads = 8
                while input_nc % num_heads != 0 and num_heads > 1:
                    num_heads //= 2
                if num_heads < 1:
                    num_heads = 1
                self.attention = TransformerBlock(in_channels=input_nc, num_heads=num_heads)
                print(f"✓ Adding TRANSFORMER attention to block: depth={depth}")
            else:
                self.attention = None
        else:
            self.attention = None
            if use_trans > 0 and not outermost:
                print(f"✗ SKIPPING attention for block: depth={depth} < threshold={transformer_depth_threshold}")

        if self.outermost:
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(
                uprelu,
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )
        elif self.innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            downseq = [downrelu, downconv, downnorm]
            upseq = [uprelu, upconv, upnorm]
            if use_dropout:
                upseq += [nn.Dropout(0.5)]
            self.down = nn.Sequential(*downseq)
            self.up = nn.Sequential(*upseq)

    def forward(self, x, feats=None):
        if self.outermost:
            out_down = self.down(x)
            if self.submodule is not None:
                if feats is None:
                    out_sub, sub_feats = self.submodule(out_down, feats=None)
                else:
                    out_sub, sub_feats = self.submodule(out_down, feats=feats)
            else:
                out_sub, sub_feats = out_down, []
            out_up = self.up(out_sub)
            return out_up, sub_feats

        elif self.innermost:
            # Apply attention to innermost block too
            x_processed = x
            if self.attention is not None:
                x_processed = self.attention(x)
            
            if feats is None:
                out_down = self.down(x_processed)
                out_up = self.up(out_down)
                sub_feats = [out_down]
                return torch.cat([x_processed, out_up], 1), sub_feats
            else:
                out_down = feats[0]
                out_up = self.up(out_down)
                return torch.cat([x_processed, out_up], 1), feats[1:]

        else:
            # Apply attention to the skip connection input before processing
            x_processed = x
            if self.attention is not None:
                x_processed = self.attention(x)
            
            # middle blocks
            if feats is None:
                out_down = self.down(x_processed)
                out_sub, sub_feats = self.submodule(out_down, feats=None)
                out_up = self.up(out_sub)
                sub_feats.append(out_down)
                return torch.cat([x_processed, out_up], 1), sub_feats
            else:
                out_down = feats[-1]
                feats = feats[:-1]
                out_sub, feats = self.submodule(out_down, feats=feats)
                out_up = self.up(out_sub)
                return torch.cat([x_processed, out_up], 1), feats

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_nll=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        # input_nc * 0.5 for probabilistic model
        print('input_nc: ', input_nc)
        if use_nll:
            input_nc = (input_nc - 1) * 0.5 + 1
            input_nc = int(input_nc)
        print('input_nc: ', input_nc)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class Spectral_Atten(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        self.num_heads = heads
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x_in):
        """
        x_in: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x_in.shape
        q_in = self.q_dwconv(self.to_q(x_in))
        k_in = self.k_dwconv(self.to_k(x_in))
        v_in = self.v_dwconv(self.to_v(x_in))

        q = rearrange(q_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
  
        q = F.normalize(q, dim=-1, p=2) 
        k = F.normalize(k, dim=-1, p=2)
        atten = (q @ k.transpose(-2, -1)) * self.rescale
        atten = atten.softmax(dim=-1)
        out = (atten @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)

        return out

def count_parameters(model):
    """Count the total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model, model_name="Model"):
    """Print parameter count and attention modules"""
    total_params = count_parameters(model)
    print(f"\n{model_name} total parameters: {total_params:,}")
    
    # Count attention modules with detailed breakdown
    attention_count = 0
    transformer_count = 0
    attention_params = 0
    all_modules = 0
    
    print("\nModule breakdown:")
    for name, module in model.named_modules():
        all_modules += 1
        if isinstance(module, Spectral_Atten):
            attention_count += 1
            module_params = count_parameters(module)
            attention_params += module_params
            print(f"  ✓ Spectral Attention module {name}: {module_params:,} parameters")
        elif isinstance(module, TransformerBlock):
            transformer_count += 1
            module_params = count_parameters(module)
            attention_params += module_params
            print(f"  ✓ Transformer module {name}: {module_params:,} parameters")
        elif isinstance(module, UnetSkipConnectionBlock):
            has_attention = hasattr(module, 'attention') and module.attention is not None
            attention_type = "None"
            if has_attention:
                if isinstance(module.attention, Spectral_Atten):
                    attention_type = "Spectral"
                elif isinstance(module.attention, TransformerBlock):
                    attention_type = "Transformer"
            print(f"  UnetBlock {name}: attention={attention_type}")
    
    print(f"\nTotal modules scanned: {all_modules}")
    print(f"Total spectral attention modules: {attention_count}")
    print(f"Total transformer modules: {transformer_count}")
    print(f"Total attention parameters: {attention_params:,}")
    print(f"Non-attention parameters: {total_params - attention_params:,}")
    return total_params



