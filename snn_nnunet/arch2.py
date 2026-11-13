"""Three patch-embedding/VSS stages and the standard two-skip decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from snn_nnunet.arch2_config import arch2_options
from snn_nnunet.model import ConvBlock, DeconvBlock
from snn_nnunet.precision import FP32Module, fp32_forward
from snn_nnunet.spike_neurons import PLIFNode
from snn_nnunet.spikmamba import Spiking2DPatchEmbedding, SpikMambaBlock
from snn_nnunet import surrogate


class Arch2EncoderStage(nn.Module):
    """Non-overlapping patch embedding, spatial mixer, GN and PLIF or GELU."""
    def __init__(self, in_channels, channels, config, selective_scan=None):
        super().__init__()
        patch = config['patch_size']
        self.patch_embed = Spiking2DPatchEmbedding(
            in_channels, channels, image_size=(patch, patch), patch_size=patch,
            max_time_steps=4096)
        self.mixer = SpikMambaBlock(
            dim=channels, selective_scan=selective_scan, max_time_steps=4096,
            init_tau=config['neurons']['init_tau'],
            patch_embedding_spiking=config['encoder']['patch_embedding_spiking'],
            dwconv2d_spiking=config['encoder']['dwconv2d_spiking'], **config['ssm'])
        self.post_norm = nn.GroupNorm(1, channels)
        self.post_plif = (PLIFNode(init_tau=config['neurons']['init_tau'])
                          if config['encoder']['vss_output_spiking'] else nn.GELU())
        self.dropout = config['dropout']

    def forward(self, x, time_step):
        x = self.patch_embed(x, time_step)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.mixer(tokens, time_step, spatial_shape=(height, width))
        x = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.post_norm(x)
        x = self.post_plif(x, time_step)[0] if isinstance(self.post_plif, PLIFNode) else self.post_plif(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class Arch2(FP32Module):
    """Process [B,T,4,H,W] slice sequences into raw [B,3,T,H,W] WT/TC/ET logits."""
    def __init__(self, out_channels=3, selective_scan=None, **options):
        super().__init__()
        if type(out_channels) is not int or out_channels != 3:
            raise ValueError('Arch2 requires three output logits [WT, TC, ET]')
        self.config = arch2_options(options)
        config = self.config
        c1, c2, c3 = config['encoder']['channels']
        d1, d2, d3 = config['decoder']['channels']
        patch = config['patch_size']
        self.encoder_scale = patch ** 3
        self.encoder = nn.ModuleList([
            Arch2EncoderStage(cin, cout, config, selective_scan)
            for cin, cout in ((4, c1), (c1, c2), (c2, c3))])
        decoder_options = dict(dropout=config['dropout'], init_tau=config['neurons']['init_tau'],
                               output_spiking=config['decoder']['spiking'])
        self.deconv_block1 = DeconvBlock(c3, d1, patch, patch, **decoder_options)
        self.deconv1_conv = ConvBlock(d1, d1, padding=1, **decoder_options)
        self.concat1_conv = ConvBlock(d1 + c2, d1, padding=1, **decoder_options)
        self.deconv_block2 = DeconvBlock(d1, d2, patch, patch, **decoder_options)
        self.deconv2_conv = ConvBlock(d2, d2, padding=1, **decoder_options)
        self.concat2_conv = ConvBlock(d2 + c1, d2, padding=1, **decoder_options)
        self.deconv_block3 = DeconvBlock(d2, d3, patch, patch, **decoder_options)
        self.deconv3_conv = ConvBlock(d3, d3, padding=1, **decoder_options)
        self.class_conv = ConvBlock(d3, out_channels, padding=1, dropout=0.,
                                    normalization=False, spiking=False, activation=False)
        # The linear classifier must not register unused normalization parameters.
        self.class_conv.norm = nn.Identity()
        neuron_options = config['neurons']
        for module in self.modules():
            if isinstance(module, PLIFNode):
                module.v_threshold = neuron_options['v_threshold']
                module.v_reset = neuron_options['v_reset']
                module.init_v = neuron_options['v_reset'] or 0.
                module.detach_reset = neuron_options['detach_reset']
                module.surrogate_function = surrogate.ATan(alpha=neuron_options['surrogate_alpha'])
        self.deep_supervision_enabled = config['deep_supervision']['enabled']
        self.deep_supervision_resolution = config['deep_supervision']['resolution']
        self.auxiliary_heads = nn.ModuleList()
        if self.deep_supervision_enabled:
            # Match the baseline: extra heads cannot perturb common weights or RNG.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                self.auxiliary_heads.extend((nn.Conv2d(d2, out_channels, 1),
                                             nn.Conv2d(d1, out_channels, 1)))

    @fp32_forward
    def forward(self, x, t0=0, *, return_auxiliary=False):
        if type(return_auxiliary) is not bool:
            raise ValueError('return_auxiliary must be boolean')
        if return_auxiliary and not self.deep_supervision_enabled:
            raise ValueError('return_auxiliary requires deep_supervision.enabled')
        if x.ndim != 5 or x.shape[2] != 4:
            raise ValueError('Arch2 expects [B,T,4,H,W] inputs')
        if any(size <= 0 for size in x.shape):
            raise ValueError('Arch2 input dimensions must be nonempty')
        if type(t0) is not int or t0 < 0 or t0 + x.shape[1] > 4096:
            raise ValueError('Arch2 slice indices must be in [0, 4096)')
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % self.encoder_scale, (-width) % self.encoder_scale
        logits = []
        auxiliary = ([], []) if return_auxiliary else None
        for index in range(x.shape[1]):
            time_step = t0 + index
            frame = F.pad(x[:, index], (0, pad_w, 0, pad_h))
            skip1 = self.encoder[0](frame, time_step)
            skip2 = self.encoder[1](skip1, time_step)
            features = self.encoder[2](skip2, time_step)
            features = self.deconv1_conv(self.deconv_block1(features, time_step), time_step)
            features = self.concat1_conv(torch.cat((skip2, features), dim=1), time_step)
            if auxiliary is not None:
                auxiliary[1].append(self.auxiliary_heads[1](features))
            features = self.deconv2_conv(self.deconv_block2(features, time_step), time_step)
            features = self.concat2_conv(torch.cat((skip1, features), dim=1), time_step)
            if auxiliary is not None:
                auxiliary[0].append(self.auxiliary_heads[0](features))
            features = self.deconv3_conv(self.deconv_block3(features, time_step), time_step)
            logits.append(self.class_conv(features, time_step)[..., :height, :width])
        main = torch.stack(logits, dim=2)
        if auxiliary is None:
            return main
        if self.deep_supervision_resolution == 'native':
            return (main, *(torch.stack(slices, dim=2) for slices in auxiliary))
        outputs = [main]
        for slices in auxiliary:
            sequence = torch.stack(slices, dim=1)
            batch, steps, channels = sequence.shape[:3]
            resized = F.interpolate(sequence.flatten(0, 1), size=(height+pad_h, width+pad_w),
                                    mode='bilinear', align_corners=False)[..., :height, :width]
            outputs.append(resized.reshape(batch, steps, channels, height, width).transpose(1, 2))
        return tuple(outputs)

    def detach_states(self):
        for module in self.modules():
            if isinstance(module, PLIFNode):
                module.detach()
