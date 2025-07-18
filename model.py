# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPositionalEmbedding(nn.Module):
    """
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1000):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension)
        even_indices = torch.arange(0, self.dimension, 2)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep):
        self.pe_matrix = self.pe_matrix.to(timestep.device)
        return self.pe_matrix[timestep].to(timestep.device)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding)

    def forward(self, input_tensor):
        return self.conv(input_tensor)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0):
        super(UpsampleBlock, self).__init__()
        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, input_tensor):
        x = F.interpolate(input_tensor, scale_factor=self.scale, mode="bilinear", align_corners=True)
        return self.conv(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_channels=None, num_groups=8):
        super(ResNetBlock, self).__init__()
        self.time_embedding_projectile = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_channels, out_channels))
            if time_emb_channels
            else None
        )
        self.block1 = ConvBlock(in_channels, out_channels, groups=num_groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=num_groups)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding=None):
        input_tensor = x
        h = self.block1(x)
        if self.time_embedding_projectile:
            time_emb = self.time_embedding_projectile(time_embedding)
            x = time_emb[:, :, None, None] + h
        x = self.block2(x)
        return x + self.residual_conv(input_tensor)

class SelfAttentionBlock(nn.Module):
    def __init__(self, num_heads, in_channels, num_groups=32, embedding_dim=256):
        super(SelfAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.d_model = embedding_dim
        self.d_keys = embedding_dim // num_heads
        self.d_values = embedding_dim // num_heads
        self.query_projection = nn.Linear(in_channels, embedding_dim)
        self.key_projection = nn.Linear(in_channels, embedding_dim)
        self.value_projection = nn.Linear(in_channels, embedding_dim)
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=num_groups)

    def split_features_for_heads(self, tensor):
        batch, hw, emb_dim = tensor.shape
        channels_per_head = emb_dim // self.num_heads
        heads_splitted_tensor = torch.stack(torch.split(tensor, split_size_or_sections=channels_per_head, dim=-1), 1)
        return heads_splitted_tensor

    def forward(self, input_tensor):
        batch, features, h, w = input_tensor.shape
        x = input_tensor.view(batch, features, h * w).transpose(1, 2)
        queries = self.split_features_for_heads(self.query_projection(x))
        keys = self.split_features_for_heads(self.key_projection(x))
        values = self.split_features_for_heads(self.value_projection(x))
        
        scale = self.d_keys ** -0.5
        attention_scores = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1)
        attention_scores = torch.matmul(attention_scores, values)
        
        concatenated_heads_attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous().view(batch, h * w, self.d_model)
        
        linear_projection = self.final_projection(concatenated_heads_attention_scores).transpose(-1, -2).reshape(batch, self.d_model, h, w)
        
        return self.norm(linear_projection + input_tensor)

class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, downsample=True):
        super(ConvDownBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_c, out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.downsample = DownsampleBlock(out_channels, out_channels, 2, 1) if downsample else None

    def forward(self, x, time_embedding):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.downsample:
            x = self.downsample(x)
        return x

class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, upsample=True):
        super(ConvUpBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_c, out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.upsample = UpsampleBlock(out_channels, out_channels) if upsample else None

    def forward(self, x, time_embedding):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.upsample:
            x = self.upsample(x)
        return x

class AttentionDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, num_att_heads, downsample=True):
        super(AttentionDownBlock, self).__init__()
        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_c, out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
            attention_blocks.append(SelfAttentionBlock(num_att_heads, out_channels, num_groups=num_groups, embedding_dim=out_channels))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.downsample = DownsampleBlock(out_channels, out_channels, 2, 1) if downsample else None

    def forward(self, x, time_embedding):
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.downsample:
            x = self.downsample(x)
        return x

class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, num_att_heads, upsample=True):
        super(AttentionUpBlock, self).__init__()
        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_c, out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
            attention_blocks.append(SelfAttentionBlock(num_att_heads, out_channels, num_groups=num_groups, embedding_dim=out_channels))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.upsample = UpsampleBlock(out_channels, out_channels) if upsample else None

    def forward(self, x, time_embedding):
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.upsample:
            x = self.upsample(x)
        return x

class UNet(nn.Module):
    def __init__(self, image_size=256, input_channels=3):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )
        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(128, 128, 2, 128 * 4, 32),
            ConvDownBlock(128, 128, 2, 128 * 4, 32),
            ConvDownBlock(128, 256, 2, 128 * 4, 32),
            AttentionDownBlock(256, 256, 2, 128 * 4, 32, 4),
            ConvDownBlock(256, 512, 2, 128 * 4, 32)
        ])
        self.bottleneck = AttentionDownBlock(512, 512, 2, 128 * 4, 32, 4, downsample=False)
        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(512 + 512, 512, 2, 128 * 4, 32),
            AttentionUpBlock(512 + 256, 256, 2, 128 * 4, 32, 4),
            ConvUpBlock(256 + 256, 256, 2, 128 * 4, 32),
            ConvUpBlock(256 + 128, 128, 2, 128 * 4, 32),
            ConvUpBlock(128 + 128, 128, 2, 128 * 4, 32)
        ])
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 3, 3, padding=1)
        )

    def forward(self, x, time):
        time_encoded = self.positional_encoding(time)
        initial_x = self.initial_conv(x)
        skips = [initial_x]
        
        for block in self.downsample_blocks:
            initial_x = block(initial_x, time_encoded)
            skips.append(initial_x)
            
        initial_x = self.bottleneck(initial_x, time_encoded)
        
        for i, block in enumerate(self.upsample_blocks):
            initial_x = torch.cat([initial_x, skips[-(i+1)]], dim=1)
            initial_x = block(initial_x, time_encoded)
            
        initial_x = torch.cat([initial_x, skips[0]], dim=1)
        return self.output_conv(initial_x)

