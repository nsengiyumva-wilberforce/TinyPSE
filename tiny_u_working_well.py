from typing import List, Tuple
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

from libs.conv_stft import ConvSTFT, ConviSTFT


def autopad(k, p=None, d=1):  
    '''
    Automatically computes padding for 1D integers or 2D (freq, time) tuples.
    '''
    if isinstance(k, int):
        k = (k, k)
    if isinstance(d, int):
        d = (d, d)
    if isinstance(p, int):
        p = (p, p)

    k_eff = [di * (ki - 1) + 1 for ki, di in zip(k, d)]
    if p is None:
        p = [xi // 2 for xi in k_eff]
    return tuple(p)


class Conv(nn.Module):
    """Spectrogram convolution with LayerNorm."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        padding = autopad(k, p, d)
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution for Spectrograms."""
    def __init__(self, c1, c2, k=3, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

    
class CMRF(nn.Module):
    """CMRF Module adapted for Spectrogram Feature Extraction."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.N         = N
        self.c         = int(c2 * e / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2   = Conv(c2//2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=(3, 3), act=False) for _ in range(N-1))

    def forward(self, x):
        x_residual = x
        x          = self.pwconv1(x)

        x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0]       = x[0] +  x[1] 
        x.pop(1)
        
        y          = th.cat(x, dim=1) 
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y
    

class SelfAttentivePooling2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    @th.compile
    def forward(self, x):
        B, C, F, T = x.shape
        attn_scores = self.attn_conv(x)
        attn_weights = self.softmax(attn_scores.view(B, 1, -1))
        flat_x = x.view(B, C, -1)
        pooled = th.bmm(flat_x, attn_weights.transpose(1, 2))
        return pooled.view(B, C, 1, 1)


class LCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(LCA, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=(1,2), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(inter_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=(2,1), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(channels),         
         )

        self.global_att = nn.Sequential(
            SelfAttentivePooling2d(channels),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),         
        )     

        self.sigmoid = nn.Sigmoid()

    @th.compile
    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class IFI(nn.Module):
    def __init__(self, channels=64, r=4):
        super(IFI, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),         
        )

        self.global_att = nn.Sequential(
            SelfAttentivePooling2d(channels),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),         
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),         
        )
        
        self.global_att2 = nn.Sequential(
            SelfAttentivePooling2d(channels),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),         
        )

        self.sigmoid = nn.Sigmoid()

    @th.compile
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

    
class Conv2dBlock(nn.Module):
    def __init__(self, in_dims: int = 16, out_dims: int = 32, kernel_size: Tuple[int] = (3, 3), stride: Tuple[int] = (1, 1), padding: Tuple[int] = (1, 1)) -> None:
        super(Conv2dBlock, self).__init__() 
        self.conv2d = nn.Conv2d(in_dims, out_dims, kernel_size, stride, padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv2d(x)
        x = self.elu(x)
        return self.norm(x)

    
class DenseBlock(nn.Module):
    def __init__(self, in_dims, out_dims, mode = "enc", **kargs):
        super(DenseBlock, self).__init__()
        if mode not in ["enc", "dec"]:
            raise RuntimeError("The mode option must be 'enc' or 'dec'!")
            
        n = 1 if mode == "enc" else 2
        self.conv1 = Conv2dBlock(in_dims=in_dims*n, out_dims=in_dims, **kargs)
        self.conv2 = Conv2dBlock(in_dims=in_dims*(n+1), out_dims=in_dims, **kargs)
        self.conv3 = Conv2dBlock(in_dims=in_dims*(n+2), out_dims=in_dims, **kargs)
        self.conv4 = Conv2dBlock(in_dims=in_dims*(n+3), out_dims=in_dims, **kargs)
        self.conv5 = Conv2dBlock(in_dims=in_dims*(n+4), out_dims=out_dims, **kargs)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(th.cat([x, y1], 1))
        y3 = self.conv3(th.cat([x, y1, y2], 1))
        y4 = self.conv4(th.cat([x, y1, y2, y3], 1))
        y5 = self.conv5(th.cat([x, y1, y2, y3, y4], 1))
        return y5


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.cmrf       = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x

    
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.cmrf      = CMRF(in_channels, out_channels)
        self.upsample  = F.interpolate

    @th.compile
    def forward(self, x, skip_connection):
        x = self.upsample(x, size=skip_connection.shape[-2:], mode='bicubic')
        x = th.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x
    
    



class TACT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TACT, self).__init__()
        # Single unified input projection to save memory bandwidth
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Combined depthwise convolutions using groups to process branches faster
        # Keeping your 3x3 and 5x5 concept but structured sequentially/efficiently
        self.conv_f = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.conv_t = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=False)
        
        # Lightweight Normalization 
        self.bn = nn.BatchNorm2d(out_channels)
        self.gate_activation = nn.Sigmoid()

        # Learnable scalars simplified to scalar constants for cleaner fusion
        self.alpha1 = nn.Parameter(th.tensor(1.0))
        self.alpha2 = nn.Parameter(th.tensor(1.0))

    @th.compile
    def forward(self, x):
        # 1. Project to target channel space once
        x_proj = self.proj(x)
        
        # 2. Extract features efficiently
        f_feat = self.conv_f(x_proj)
        t_feat = self.conv_t(x_proj)
        
        # 3. Streamlined cross-branch interaction (retains your concept without redundant BN layers)
        branch1 = (self.alpha1 * f_feat) + self.conv_t(f_feat)
        branch2 = (self.alpha2 * t_feat) + self.conv_f(t_feat)
        
        # 4. Single fused normalization & residual connection
        combined = self.bn(branch1 + branch2)
        overall_features = combined + x_proj  # Replaced costly redundant conv1x1(x) with x_proj
        
        # 5. Gated attention output
        return x_proj * self.gate_activation(overall_features)
        
    

class TinyPSE(nn.Module):
    def __init__(self, in_channels=16, num_classes=2, win_len: int = 256, win_inc: int = 64, fft_len: int = 256, win_type: str = "sqrthann", kernel_size: Tuple[int] = (3, 3), stride1: Tuple[int] = (1, 1), stride2: Tuple[int] = (1, 2), paddings: Tuple[int] = (1, 0), output_padding: Tuple[int] = (0, 0), causal: bool = False, pool_size: Tuple[int] = (4, 8, 16, 32), num_spks: int = 1):
        super(TinyPSE, self).__init__()
        in_filters      = [192, 384, 768, 1024]
        out_filters     = [64, 128, 256, 512]

        self.fft_len = fft_len
        self.num_spks = num_spks
        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        self.softmax = nn.Softmax(dim=-2)
        # self.ifi = IFI(channels=2, r=1/32)
        self.upconv1 = nn.Conv2d(6, 64, 1, 1, 0)
        self.lca = LCA(64)
        self.conv2d = nn.Conv2d(64, 16, (1, 3), stride=1, padding=(0, 1))
        self.relu = nn.ReLU() 

        self.conv3x3 = Conv2dBlock(in_dims=16, out_dims=16, kernel_size=(1, 3), stride=stride1, padding=(0, 1))
        self.conv5x5 = Conv2dBlock(in_dims=16, out_dims=16, kernel_size=(1, 5), stride=stride1, padding=(0, 2))
        self.conv1x1 = Conv2dBlock(in_dims=16, out_dims=16, kernel_size=(1, 1), stride=stride1, padding=(0, 0))
        
        self.bn1 = nn.BatchNorm2d(16) 

        self.encoder1   = UNetEncoder(in_channels, 64)
        self.encoder2   = UNetEncoder(64, 128)
        self.encoder3   = UNetEncoder(128, 256)
        self.encoder4   = UNetEncoder(256, 512)
        

        self.decoder4   = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3   = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2   = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1   = UNetDecoder(in_filters[0], out_filters[0])
        self.tact1 = TACT(in_channels=out_filters[0], out_channels=out_filters[0])  
        self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type, 'complex')

    @th.compile
    def FeaCompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = th.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = th.abs(complex_spectrum).unsqueeze(1) ** factor
        phase = th.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * th.cos(phase)
        imag = magnitude * th.sin(phase)
        output = th.cat((real, imag), dim=1)
        return output  

    def wav2spec(self, x: th.Tensor, mags: bool = False) -> th.Tensor:
        assert x.dim() == 2  
        specs = self.stft(x)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec = th.stack([real,imag], 1) 
        if mags:
            return th.sqrt(real**2+imag**2+1e-8)
        else:
            return spec     

    @th.compile
    def FeaDecompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = th.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = th.abs(complex_spectrum).unsqueeze(1) ** (1 / factor)
        phase = th.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * th.cos(phase)
        imag = magnitude * th.sin(phase)
        output = th.cat((real, imag), dim=1)
        return output


    @th.compile
    def ComputeSimilarity(self, input, enrollment, eps=1e-8):
        # 1. Construct PyTorch Complex Tensors
        e_c = th.complex(enrollment[:, 0], enrollment[:, 1]) # Shape: [B, F, T_e]
        i_c = th.complex(input[:, 0], input[:, 1])           # Shape: [B, F, T_i]
        
        # 2. The Complex Conjugate Transpose (Hermitian)
        # .mH is the mathematical adjoint (transpose + imaginary conjugate). 
        # THIS is required for valid complex geometry.
        e_H = e_c.mH # Shape: [B, T_e, F]
        
        # 3. Hermitian Inner Product (Complex Cross-Covariance)
        # Measures how well every mixture frame aligns with every clean enrollment frame.
        # If the phase is misaligned, the complex math naturally cancels it out!
        complex_affinity = e_H @ i_c # Shape: [B, T_e, T_i]
        
        # 4. Extract the absolute magnitude of the complex affinity
        # This gives us a real-valued score of geometric similarity
        affinity_mag = complex_affinity.abs()
        
        # 5. Temperature Scaling (Crucial for deep learning)
        # Dividing by sqrt(F) prevents the Softmax from collapsing into one-hot vectors
        F_dim = e_c.shape[1]
        scaled_affinity = affinity_mag / (F_dim ** 0.5)
        
        # 6. Generate Attention Probabilities
        # dim=-2 ensures that for every mixture frame, we pull a valid distribution of clean frames
        att_weights = th.softmax(scaled_affinity, dim=-2) 
        
        # 7. Dictionary Synthesis
        # Cast weights back to complex so PyTorch can matrix multiply them
        att_weights = att_weights.to(e_c.dtype)
        
        # We reconstruct the target speech by multiplying the clean enrollment dictionary 
        # by the attention weights. Noise is mathematically left behind.
        out_c = e_c @ att_weights # Shape: [B, F, T_i]
        
        # 8. Recombine real and imaginary parts for the U-Net
        return th.stack([out_c.real, out_c.imag], dim=1)
    
    def sep(self, spec: th.Tensor) -> List[th.Tensor]:
        B, N, F, T = spec.shape
        est = th.chunk(spec, 2, 1)      
        est = th.cat(est, 2).reshape(B, -1, T)      
        return th.squeeze(self.istft(est))   
        
    def forward(self, mix: th.Tensor, enrollment: th.Tensor) -> th.Tensor:
        # 1. Process standard mixture spectrogram entirely in batch
        mix_spec = self.wav2spec(mix, False)
        mix_spec_change = self.FeaCompression(mix_spec) 
        
        aux = self.wav2spec(enrollment, False)
        aux_drc = self.FeaCompression(aux)
        
        similarity = self.ComputeSimilarity(mix_spec_change, aux_drc)  # Yields perfectly scaled [B, 2, F, T]
        
        aux_drc = F.adaptive_avg_pool2d(aux_drc, (aux_drc.shape[-2], 1)) 
        # similarity = self.ifi(similarity, aux_drc.expand(-1, -1, -1, similarity.shape[-1]))
        similarity = th.cat((similarity, aux_drc.expand(-1, -1, -1, similarity.shape[-1])), dim=1)  # Concatenate along channel dimension
        
        fus = th.cat((mix_spec_change, similarity), dim=1) 
        fus = self.upconv1(fus)
        fus = self.lca(fus)
        
        fus = fus.permute(0, 1, 3, 2)
        out = self.relu(self.conv2d(fus))     

        x = out
        short_x = self.conv3x3(x)

        long_x = self.conv5x5(x)
        pw_x = self.conv1x1(x)
        x = short_x + long_x + pw_x
        
        x = self.bn1(x)
        x = x + x
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x) 

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.tact1(x)

        x = x.permute(0, 1, 3, 2) 
        x = self.FeaDecompression(x) 
        x = self.sep(x) 
        return x



