from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from improved_diffusion.RRDB import RRDBNet
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class ResidualBlock(nn.Module):
    """用于Shape Stream的残差块"""
    def __init__(self, channels, dims=2):
        super().__init__()
        self.conv1 = conv_nd(dims, channels, channels, 3, padding=1)
        self.norm1 = normalization(channels)
        self.conv2 = conv_nd(dims, channels, channels, 3, padding=1)
        self.norm2 = normalization(channels)
        self.relu = SiLU()
        
    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)

class BoundaryDetectionStream(nn.Module):
    """边界检测分支"""
    def __init__(self, in_channels=3, base_channels=64, dims=2):   #in_channels=3/1
        super().__init__()
        
        # Stem: 256 -> 128
        self.stem = nn.Sequential(
            conv_nd(dims, in_channels, base_channels, 7, stride=2, padding=3),
            normalization(base_channels),
            SiLU()
        )
        
        # Stage 1: 128x128, 64 channels
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels, dims),
            ResidualBlock(base_channels, dims)
        )
        
        # Stage 2: 128 -> 64, 128 channels  
        self.stage2 = nn.Sequential(
            conv_nd(dims, base_channels, base_channels*2, 3, stride=2, padding=1),
            normalization(base_channels*2),
            SiLU(),
            ResidualBlock(base_channels*2, dims),
            ResidualBlock(base_channels*2, dims)
        )
        
        # Stage 3: 64 -> 32, 256 channels
        self.stage3 = nn.Sequential(
            conv_nd(dims, base_channels*2, base_channels*4, 3, stride=2, padding=1),
            normalization(base_channels*4),
            SiLU(),
            ResidualBlock(base_channels*4, dims),
            ResidualBlock(base_channels*4, dims)
        )
        
        # Stage 4: 32 -> 16, 512 channels
        self.stage4 = nn.Sequential(
            conv_nd(dims, base_channels*4, base_channels*8, 3, stride=2, padding=1),
            normalization(base_channels*8),
            SiLU(),
            ResidualBlock(base_channels*8, dims),
            ResidualBlock(base_channels*8, dims)
        )
        
        # 边界预测头
        self.boundary_heads = nn.ModuleList([
            conv_nd(dims, base_channels, 1, 1),        # 128分辨率
            conv_nd(dims, base_channels*2, 1, 1),      # 64分辨率  
            conv_nd(dims, base_channels*4, 1, 1),      # 32分辨率
            conv_nd(dims, base_channels*8, 1, 1),      # 16分辨率
        ])
        
    def forward(self, x):
        stem_out = self.stem(x)         # (B, 64, 128, 128)
        stage1_out = self.stage1(stem_out)    # (B, 64, 128, 128)
        stage2_out = self.stage2(stage1_out)  # (B, 128, 64, 64)
        stage3_out = self.stage3(stage2_out)  # (B, 256, 32, 32)  
        stage4_out = self.stage4(stage3_out)  # (B, 512, 16, 16)
        
        boundary_features = [stage1_out, stage2_out, stage3_out, stage4_out]
        
        # 边界预测
        boundary_predictions = []
        for stage_feat, head in zip(boundary_features, self.boundary_heads):
            boundary_pred = head(stage_feat)
            boundary_predictions.append(boundary_pred)
        
        return {
            'boundary_features': boundary_features,
            'boundary_predictions': boundary_predictions,
        }
    
class GatedBoundaryFusion(nn.Module):
    """门控边界融合模块"""
    def __init__(self, main_channels, boundary_channels, dims=2):
        super().__init__()
        
        self.boundary_align = conv_nd(dims, boundary_channels, main_channels, 1)
        
        self.gate_network = nn.Sequential(
            conv_nd(dims, main_channels * 2, main_channels, 3, padding=1),
            normalization(main_channels),
            SiLU(),
            conv_nd(dims, main_channels, main_channels, 3, padding=1),
            normalization(main_channels),
            nn.Sigmoid()
        )
        
        self.fusion_network = nn.Sequential(
            conv_nd(dims, main_channels * 2, main_channels, 3, padding=1),
            normalization(main_channels),
            SiLU(),
            conv_nd(dims, main_channels, main_channels, 3, padding=1),
            normalization(main_channels)
        )
        
    def forward(self, main_feat, boundary_feat):
        boundary_aligned = self.boundary_align(boundary_feat)
        
        if boundary_aligned.shape[2:] != main_feat.shape[2:]:
            boundary_aligned = F.interpolate(
                boundary_aligned, 
                size=main_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        combined = th.cat([main_feat, boundary_aligned], dim=1)
        gate = self.gate_network(combined)
        fused_feat = self.fusion_network(combined)
        
        return main_feat + gate * fused_feat

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        # 初始化父类
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        # up=False,
        # down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self, 
            channels, 
            num_heads=1, 
            use_checkpoint=False):
        super().__init__()
        # print(f"AttentionBlock init - channels: {channels}, num_heads: {num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape   # b是batch_size 为4， c是通道数为128， spatial是空间维度
        x = x.reshape(b, c, -1)    # [B, C,H*W]
        # print(f"AttentionBlock _forward - num_heads: {self.num_heads}, shape: {x.shape}")
        qkv = self.qkv(self.norm(x))   # [B, 3*C, H*W]
        # print(f"Before final reshape - qkv shape: {qkv.shape}, b: {b}, num_heads: {self.num_heads}")
        # 在此处手动进行多头分割
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        # 调用不含多头处理的attention类
        h = self.attention(qkv)
         # 在此处手动合并多头结果
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3    # ch 为当前通道数
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])

class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,  # maks输入通道
        model_channels,   # 基础通道数
        out_channels, # 输出通道数（向量场）
        num_res_blocks, 
        attention_resolutions, # 注意力层分辨率，用于跳跃连接，如[16]，表示第16层有注意力层
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        rrdb_blocks=3,
        rrdb_out_channels=None,
        enable_boundary_fusion=True,  # 新增参数
        boundary_fusion_scales=[128, 64, 32, 16]  # 新增参数：选择在哪些分辨率融合 
        
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        # 初始化父类
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # print(f"UNetModel init - num_heads: {num_heads}, num_heads_upsample: {num_heads_upsample}")
        self.rrdb_blocks = rrdb_blocks
        self.enable_boundary_fusion = enable_boundary_fusion
        self.boundary_fusion_scales = boundary_fusion_scales
        # 简化的时间嵌入
        time_embed_dim = model_channels * 4    # 512维特征
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # 条件图像编码器
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        self.rrdb = RRDBNet(nb=rrdb_blocks, out_nc=model_channels)

        if enable_boundary_fusion:
            self.boundary_stream = BoundaryDetectionStream(in_channels=3, dims=dims)   # in_channels=1  or  in_channels=3
            
            # 门控融合模块 - 对应你的channel_mult
            self.boundary_fusions = nn.ModuleDict()
            fusion_configs = {
                128: (model_channels, 64),      # level 0-1: 128ch, boundary 64ch
                64: (model_channels, 128),    # level 2: 256ch, boundary 128ch  
                32: (model_channels*2, 256),    # level 3: 256ch, boundary 256ch
                16: (model_channels*2, 512),    # level 4: 512ch, boundary 512ch
            }
            
            for scale in boundary_fusion_scales:
                if scale in fusion_configs:
                    main_ch, boundary_ch = fusion_configs[scale]
                    self.boundary_fusions[str(scale)] = GatedBoundaryFusion(
                        main_channels=main_ch, 
                        boundary_channels=boundary_ch, 
                        dims=dims
                    )
        else:
            self.boundary_stream = None
            self.boundary_fusions = None

        #  # 初始通道數
        # ch = model_channels   # ch为当前特征的通道数, 初始通道數
        # 输入blocks
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]  # 记录各层通道数用于跳跃连接
        ch = model_channels 
        ds = 1    # ds 为下采样倍率，有六层所以（1，2，4，8，16，32）
        
        # 下采样部分
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # 如果当前分辨率需要注意力机制
                if ds in attention_resolutions:    # 注意问chatgpt这个需要吗
                    print(f"Creating AttentionBlock in input_blocks - ds: {ds}, num_heads: {num_heads}")
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint, 
                            num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers)) 
                input_block_chans.append(ch)
            # 如果不是最后一级，则添加下采样层
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        # 中间block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch, 
                use_checkpoint=use_checkpoint, 
                num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )


        # 输出blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=self.num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                # self._feature_size += ch

        # 输出层 - 直接预测向量场
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.rrdb.apply(convert_module_to_f16)
        self.boundary_stream.apply(convert_module_to_f16)
        self.boundary_fusions.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.rrdb.apply(convert_module_to_f32)
        self.boundary_stream.apply(convert_module_to_f32)
        self.boundary_fusions.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, timesteps, x, y=None, conditioned_image=None):
        """
        应用模型到输入batch。
        
        Args:
            x: [N x C x ...] 张量,表示当前位置的mask
            timesteps: [N] 张量,表示时间步
            conditioned_image: [N x 3 x ...] 张量,表示条件图像
        Returns:
            [N x C x ...] 张量,表示预测的向量场
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []    # hs为保存各层特征用于跳跃连接
        boundary_outputs = None
        # 时间嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))   # t 转换为512维时间嵌入

        # 如果启用边界融合，先提取边界特征
        if self.boundary_stream is not None:
            boundary_outputs = self.boundary_stream(conditioned_image.type(self.inner_dtype))
            boundary_features = boundary_outputs['boundary_features']  # [128,64,32,16]分辨率特征
     
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        # 条件图像特征
        former_frames_features = self.rrdb(conditioned_image.type(self.inner_dtype))  #这一行与之前不一样
        current_resolution = 256  # 初始分辨率
        # 下采样路径
        h = x.type(self.inner_dtype)    # 输入mask转换为特征图
        boundary_idx = 0  # 边界特征索引

        for i, module in enumerate(self.input_blocks):
            h = module(h, emb)    # h通过每个block处理
            if i == 0:
                h = h + former_frames_features   # 第一层加入条件特征

            if (self.boundary_fusions is not None and 
                str(current_resolution) in self.boundary_fusions and boundary_outputs is not None):
                # 找到对应的边界特征
                if current_resolution == 128 and boundary_idx < len(boundary_features):
                    h = self.boundary_fusions[str(current_resolution)](h, boundary_features[0])
                elif current_resolution == 64 and boundary_idx < len(boundary_features):
                    h = self.boundary_fusions[str(current_resolution)](h, boundary_features[1])
                elif current_resolution == 32 and boundary_idx < len(boundary_features):
                    h = self.boundary_fusions[str(current_resolution)](h, boundary_features[2])
                elif current_resolution == 16 and boundary_idx < len(boundary_features):
                    h = self.boundary_fusions[str(current_resolution)](h, boundary_features[3])
            
            hs.append(h)    # 存储每层的h用于后续跳跃连接

            if hasattr(module, '__len__') and len(module) > 0:
                for layer in module:
                    if isinstance(layer, Downsample):
                        current_resolution //= 2
                        break

        # 中间处理
        h = self.middle_block(h, emb)
        
        # 上采样路径
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        # 输出向量场
        h = h.type(x.dtype)
        # 如果有边界输出，一起返回
        if boundary_outputs is not None:
            return {
                'vector_field': self.out(h),
                'boundary_predictions': boundary_outputs['boundary_predictions']
            }
        else:
            return self.out(h)
    
    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result
    
class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="nearest")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="nearest")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

