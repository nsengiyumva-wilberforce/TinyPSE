from typing import List, Tuple
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

from torch._subclasses.complex_tensor import ComplexTensor
from libs.conv_stft import ConvSTFT, ConviSTFT

th.set_float32_matmul_precision('high')


def autopad(k, p=None, d=1):
    """
    Automatically computes padding for 1D integers or 2D (freq, time) tuples.
    """
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
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SpeechDWConv(nn.Module):
    """
    Speech-optimized Depthwise Convolution.
    Factorizes Freq and Time, applies temporal dilation, and respects causality.
    """
    def __init__(self, c, k_f=3, k_t=3, dilation_t=1, causal=False, act=True):
        super().__init__()
        self.causal = causal
        
        # 1. Frequency Convolution (Localized, No dilation)
        pad_f = k_f // 2
        self.conv_f = nn.Conv2d(c, c, kernel_size=(k_f, 1), padding=(pad_f, 0), groups=c, bias=False)
        
        # 2. Time Convolution (Dilated, Causal/Non-Causal)
        self.pad_t = dilation_t * (k_t - 1)
        padding_t = 0 if causal else self.pad_t // 2
        
        self.conv_t = nn.Conv2d(c, c, kernel_size=(1, k_t), padding=(0, padding_t), 
                                dilation=(1, dilation_t), groups=c, bias=False)
        
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv_f(x)
        
        # Explicit left-padding for the time axis if causal
        if self.causal and self.pad_t > 0:
            x = F.pad(x, (self.pad_t, 0, 0, 0))
            
        x = self.conv_t(x)
        return self.act(self.bn(x))


class CMRF(nn.Module):
    """
    Speech-enhanced CMRF Module.
    Replaces 2D image-style convolutions with dilated, factorized speech convolutions.
    """
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5, causal=False):
        super().__init__()
        self.N = N
        self.c = int(c2 * e / self.N)
        self.add = shortcut and c1 == c2

        self.pwconv1 = Conv(c1, c2 // self.N, 1, 1)
        self.pwconv2 = Conv(c2 // 2, c2, 1, 1)
        
        self.m = nn.ModuleList(
            SpeechDWConv(self.c, k_f=3, k_t=3, dilation_t=(2**i), causal=causal, act=False) 
            for i in range(N - 1)
        )

        if self.add:
            self.layer_scale = nn.Parameter(th.ones(1, c2, 1, 1) * 1e-4)

    def forward(self, x):
        x_residual = x
        x = self.pwconv1(x)

        x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        
        for m in self.m:
            x.append(m(x[-1]))
            
        x[0] = x[0] + x[1]
        x.pop(1)

        y = th.cat(x, dim=1)
        y = self.pwconv2(y)

        return x_residual + y * self.layer_scale if self.add else y
    
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


class IFI(nn.Module):
    """
    Iterative Feature Integration (IFI) Module.
    """
    def __init__(self, channels=2, r=1/32):
        super(IFI, self).__init__()
        inter_channels = max(1, int(channels // r))

        self.constructive_lens = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.destructive_lens = nn.Sequential(
            SelfAttentivePooling2d(channels),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.phase_gate = nn.Sigmoid()

    def forward(self, x, residual):
        superposition = x + residual
        
        constructive = self.constructive_lens(superposition)
        destructive = self.destructive_lens(superposition)
        
        alignment = self.phase_gate(constructive + destructive)
        
        xo = x * alignment + residual * (1 - alignment)
        return xo


class SpectralPrism(nn.Module):
    """
    Spectral Prism module (Replacing Pyramidal Pooling Module).
    Instead of downsampling to fixed sizes, it treats the pool_size parameters as "dispersion factors", 
    refracting the feature map at different temporal dilation angles to capture various rhythms without losing resolution.
    """
    def __init__(self, in_channels, dispersion_factors=(4, 8, 16, 32)):
        super().__init__()
        self.refractors = nn.ModuleList()
        
        for factor in dispersion_factors:
            self.refractors.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 16, kernel_size=(3, 3), 
                              padding=(1, factor), dilation=(1, factor), bias=False),
                    nn.InstanceNorm2d(16),
                    nn.ELU(inplace=True)
                )
            )

    def forward(self, x):
        refracted_beams = [refract(x) for refract in self.refractors]
        return th.cat(refracted_beams, dim=1)





class LCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(LCA, self).__init__()
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

        self.sigmoid = nn.Sigmoid()

    @th.compile
    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei




class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, causal=False):
        super(UNetEncoder, self).__init__()
        self.cmrf = CMRF(in_channels, out_channels, causal=causal)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, causal=False):
        super(UNetDecoder, self).__init__()
        self.cmrf = CMRF(in_channels, out_channels, causal=causal)
        self.upsample = F.interpolate

    @th.compile
    def forward(self, x, skip_connection):
        x = self.upsample(x, size=skip_connection.shape[-2:], mode="bilinear", align_corners=False)
        x = th.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x

# change causal to True if you wanna trade off some accuracy for speed. 
# It will make the model causal and thus suitable for streaming applications.
class TACT(nn.Module):
    def __init__(self, in_channels, out_channels, causal: bool = False):
        super(TACT, self).__init__()
        self.causal = causal
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.conv_f = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False
        )

        self.conv_t_freq = nn.Conv2d(
            out_channels, out_channels, kernel_size=(32, 1), padding="same", groups=out_channels, bias=False
        )
        # notice that if using causal, the padding is set to "valid" and we will pad manually in the forward pass,
        # I DO NOT know why it throws an error if I set padding to "same" and use causal, but it works if I set it to "valid" and pad manually.
        if self.causal:
            self.conv_t_time = nn.Conv2d(
                out_channels, out_channels, kernel_size=(1, 32), padding="valid", groups=out_channels, bias=False
            )
        else:
            self.conv_t_time = nn.Conv2d(
                out_channels, out_channels, kernel_size=(1, 32), padding="same", groups=out_channels, bias=False
            )

        self.act = nn.GELU()
        self.channel_mix = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.gate_activation = nn.Sigmoid()

        self.alpha1 = nn.Parameter(th.tensor(1.0))
        self.alpha2 = nn.Parameter(th.tensor(1.0))

    def _apply_time_conv(self, tensor_in: th.Tensor, conv_layer: nn.Conv2d) -> th.Tensor:
        if self.causal:
            padded = F.pad(tensor_in, (31, 0, 0, 0))
            return conv_layer(padded)
        return conv_layer(tensor_in)

    @th.compile
    def forward(self, x):
        # Project input to match output channels if necessary
        x_proj = self.proj(x)

        # Apply frequency and time convolutions
        # obtain frequency features separately
        f_feat = self.act(self.conv_f(x_proj))

        # obtain temporal features separately
        t_feat = self.conv_t_freq(x_proj)
        t_feat = self.act(self._apply_time_conv(t_feat, self.conv_t_time))

        # obtain cross features by applying temporal convolution on frequency features
        cross_wide = self.conv_t_freq(f_feat)
        cross_wide = self.act(self._apply_time_conv(cross_wide, self.conv_t_time))

        branch1 = (self.alpha1 * f_feat) + cross_wide

        branch2 = (self.alpha2 * t_feat) + self.act(self.conv_f(t_feat))

        combined = self.channel_mix(branch1 + branch2)
        combined = self.bn(combined)

        overall_features = combined + x_proj
        return x_proj * self.gate_activation(overall_features)


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_dims: int = 384,
        out_dims: int = 384,
        kernel_size: int = 3,
        stride: int = 1,
        paddings: int = 1,
        dilation: int = 1,
        causal: bool = False,
    ) -> None:
        super(TCNBlock, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_dims)
        self.elu1 = nn.ELU()

        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2
            if not causal
            else (dilation * (kernel_size - 1))
        )

        self.dconv1 = nn.Conv2d(
            in_dims,
            out_dims,
            kernel_size=(1, kernel_size),
            padding=(0, dconv_pad),
            dilation=(1, dilation),
            groups=in_dims,
            bias=True,
        )

        self.norm2 = nn.InstanceNorm2d(out_dims)
        self.elu2 = nn.ELU()
        self.dconv2 = nn.Conv2d(out_dims, out_dims, 1, bias=True)

        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.elu1(self.norm1(x))
        y = self.dconv1(y)

        if self.causal and self.dconv_pad > 0:
            y = y[:, :, :, :-self.dconv_pad]

        y = self.elu2(self.norm2(y))
        y = self.dconv2(y)

        return x + y


class TinyPSE(nn.Module):
    def __init__(
        self,
        in_channels=16,
        num_classes=2,
        win_len: int = 256,
        win_inc: int = 64,
        fft_len: int = 256,
        win_type: str = "sqrthann",
        kernel_size: Tuple[int] = (3, 3),
        stride1: Tuple[int] = (1, 1),
        stride2: Tuple[int] = (1, 2),
        paddings: Tuple[int] = (1, 0),
        output_padding: Tuple[int] = (0, 0),
        causal: bool = False,
        tcn_dims: int = 384,
        tcn_blocks: int = 10,
        tcn_layers: int = 2,
        pool_size: Tuple[int] = (4, 8, 16, 32),
    ):
        super(TinyPSE, self).__init__()
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.fft_len = fft_len
        self.sim_alpha = nn.Parameter(th.tensor(1.0))
        self.sim_beta = nn.Parameter(th.tensor(1.0))
        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, "complex")
        self.softmax = nn.Softmax(dim=-2)

        # New Phase Interference IFI Module
        self.ifi = IFI(channels=2, r=1/32)

        # NOTE: Reduced in_channels to 4 because IFI outputs a 2-channel representation,
        # which is concatenated with the 2-channel mix_spec_change.
        self.upconv1 = nn.Conv2d(4, 64, 1, 1, 0)
        self.lca = LCA(64)

        self.conv2d = nn.Conv2d(64, 16, (3, 1), stride=1, padding=(1, 0))
        self.relu = nn.ReLU()

        self.encoder1 = UNetEncoder(in_channels, 64, causal=causal)
        self.encoder2 = UNetEncoder(64, 128, causal=causal)
        self.encoder3 = UNetEncoder(128, 256, causal=causal)
        self.encoder4 = UNetEncoder(256, 512, causal=causal)

        self.proj = nn.Conv2d(512, tcn_dims, 1)

        self.tcn_layers = self._build_tcn_layers(
            tcn_layers, tcn_blocks, in_dims=tcn_dims, out_dims=tcn_dims, causal=causal
        )

        self.proj_back = nn.Conv2d(tcn_dims, 512, 1)

        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3], causal=causal)
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2], causal=causal)
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1], causal=causal)
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0], causal=causal)

        self.tact1 = TACT(in_channels=out_filters[0], out_channels=out_filters[0], causal=causal)

        # Replace Pyramidal Pooling with Spectral Prism utilizing the same 'pool_size' variables
        self.prism = SpectralPrism(in_channels=out_filters[0], dispersion_factors=pool_size)

        # Dynamic mask channel calculation based on Spectral Prism output
        total_output_channels = out_filters[0] + (16 * len(pool_size))

        # Performance Upgrade: High-performance Complex Ratio Mask projection
        self.mask_conv = nn.Conv2d(total_output_channels, 2, kernel_size=1)

        self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type, "complex")

    @th.compile
    def FeaCompression(self, input_tensor, factor=0.5):
        input_change = input_tensor.float()
        # Preserving original ComplexTensor mapping
        complex_spectrum = ComplexTensor(input_change[:, 0], input_change[:, 1])

        magnitude = th.abs(complex_spectrum) ** factor
        phase = th.angle(complex_spectrum)

        real = magnitude * th.cos(phase)
        imag = magnitude * th.sin(phase)

        return th.stack([real, imag], dim=1)  # [B, 2, F, T]

    def wav2spec(self, x: th.Tensor, mags: bool = False) -> th.Tensor:
        assert x.dim() == 2
        specs = self.stft(x)
        real = specs[:, : self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1 :]
        spec = th.stack([real, imag], 1)
        if mags:
            return th.sqrt(real**2 + imag**2 + 1e-8)
        else:
            return spec

    @th.compile
    def ComputeSimilarity(self, input_tensor, enrollment, eps=1e-8):
        # Preserving original discrete real/imag slice components to ComplexTensor
        i_c = ComplexTensor(input_tensor[:, 0].float(), input_tensor[:, 1].float())  # logical: [B, F, T]
        e_c = ComplexTensor(enrollment[:, 0].float(), enrollment[:, 1].float())      # logical: [B, F, 1]

        # Explicit conjugate transpose
        e_H = th.conj(e_c.transpose(-1, -2))  # logical: [B, 1, F]

        complex_affinity = e_H @ i_c  # logical: [B, 1, T]
        affinity_mag = th.abs(complex_affinity)

        F_dim = e_c.shape[1]
        scaled_affinity = affinity_mag / (F_dim**0.5)

        att_weights = th.softmax(scaled_affinity, dim=-2)
        out_c = e_c @ att_weights  # logical: [B, F, T]

        return th.stack([out_c.real, out_c.imag], dim=1)  # [B, 2, F, T]
    
    @th.compile    
    def ComputeSimilarity_non_phase(self, input, enrollment):
        att = enrollment.transpose(-2, -1) @ input
        att = self.softmax(att)
        output = enrollment @ att

        return output

    def sep(self, spec: th.Tensor) -> List[th.Tensor]:
        B, N, Fsmall, T = spec.shape
        est = th.chunk(spec, 2, 1)
        est = th.cat(est, 2).reshape(B, -1, T)
        return th.squeeze(self.istft(est))

    def _build_tcn_blocks(self, tcn_blocks, **tcn_kargs):
        blocks = [TCNBlock(**tcn_kargs, dilation=(2**b)) for b in range(tcn_blocks)]
        return nn.Sequential(*blocks)

    def _build_tcn_layers(self, tcn_layers, tcn_blocks, **tcn_kargs):
        layers = [
            self._build_tcn_blocks(tcn_blocks, **tcn_kargs)
            for _ in range(tcn_layers)
        ]
        return nn.Sequential(*layers)

    def forward(self, mix: th.Tensor, enrollment: th.Tensor) -> th.Tensor:
        # 1. Frontend feature processing
        mix_spec = self.wav2spec(mix, False)
        mix_spec_change = self.FeaCompression(mix_spec)

        aux = self.wav2spec(enrollment, False)
        aux_drc = self.FeaCompression(aux)

        # # 2. Extract Targeted Speaker Features
        # similarity = self.ComputeSimilarity(mix_spec_change, aux_drc)  

        # 2. Extract Targeted Speaker Features
        # Get complex similarity
        sim_complex = self.ComputeSimilarity(mix_spec_change, aux_drc)  
        
        # Get real similarity and fix the shape by removing the dummy dimensions
        sim_real = self.ComputeSimilarity_non_phase(mix_spec_change, aux_drc).squeeze(0).squeeze(0)

        # COMBINE: Simple addition (or you could do (sim_complex + sim_real) * 0.5)
        similarity = (self.sim_alpha * sim_complex) + (self.sim_beta * sim_real)

        # Pass through the Phase Interference IFI Module
        aux_drc_pooled = F.adaptive_avg_pool2d(aux_drc, (aux_drc.shape[-2], 1))
        aux_drc_expanded = aux_drc_pooled.expand(-1, -1, -1, similarity.shape[-1])
        similarity = self.ifi(similarity, aux_drc_expanded)

        # Combine with Mix and run Upconv (Now dynamically 4 channels)
        fus = th.cat((mix_spec_change, similarity), dim=1)
        fus = self.upconv1(fus)
        fus = self.lca(fus)

        # 3. Backbone Flow
        out = self.relu(self.conv2d(fus))        
        x, skip1 = self.encoder1(out)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.proj(x)
        x = self.tcn_layers(x)
        x = self.proj_back(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.tact1(x)

        # 4. Feature dispersion via the Spectral Prism
        dispersed_spectrum = self.prism(x)
        x = th.cat([x, dispersed_spectrum], dim=1)

        # 5. Performance Tuning: Complex Ratio Masking (CRM)
        # Predict mask values bounded by bounded hyperbolic tangent
        mask = self.mask_conv(x)
        mask_real = th.tanh(mask[:, 0])
        mask_imag = th.tanh(mask[:, 1])

        mix_real = mix_spec[:, 0]
        mix_imag = mix_spec[:, 1]

        # Apply complex multiplication directly to linear mixture spectrum
        est_real = mix_real * mask_real - mix_imag * mask_imag
        est_imag = mix_real * mask_imag + mix_imag * mask_real

        est_spec = th.stack([est_real, est_imag], dim=1)

        # 6. Synthesis to Waveform
        return self.sep(est_spec)
