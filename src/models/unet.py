import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """Residual block for the U-Net, with optional time and class embeddings."""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, class_emb_dim=None, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.relu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        self.class_mlp = nn.Linear(class_emb_dim, out_channels) if class_emb_dim else None
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        h = self.conv1(self.relu(self.norm1(x)))
        if self.time_mlp and time_emb is not None:
            h = h + self.time_mlp(self.relu(time_emb))[:, :, None, None]
        if self.class_mlp and class_emb is not None:
            h = h + self.class_mlp(self.relu(class_emb))[:, :, None, None]
        h = self.conv2(self.dropout(self.relu(self.norm2(h))))
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """Self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        N, C, H, W = x.shape
        h_ = self.norm(x)
        q, k, v = self.qkv(h_).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(N, C, -1).transpose(-1, -2), (q, k, v))
        attn = torch.bmm(q, k.transpose(-1, -2)) * (k.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        h_ = torch.bmm(attn, v).transpose(-1, -2).reshape(N, C, H, W)
        return x + self.proj_out(h_)

class UNetModel(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0.0, num_classes=None, time_emb_dim=EMB_DIM):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions # Store for forward pass type checking
        self.model_channels = model_channels # Store for channel count checks in forward

        self.time_embed = nn.Sequential(
            PositionalEncoding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.class_embed = None
        if num_classes is not None:
            self.class_embed = nn.Sequential(
                ClassEmbedding(num_classes, time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim)
            )

        # --- Initial Convolution ---
        self.initial_conv = SequentialWithoutArgs(ConvBlock(in_channels, model_channels[0]))

        # --- Downsampling Path ---
        self.down_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        channels = model_channels[0]
        num_resolutions = len(model_channels)
        for i in range(num_resolutions):
            out_channels = model_channels[i]
            level_blocks = []
            level_input_channels = channels
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(level_input_channels, out_channels, time_emb_dim, time_emb_dim if num_classes is not None else None, dropout))
                level_input_channels = out_channels
            spatial_size = LATENT_SPATIAL_SIZE // (2**i)
            if spatial_size in attention_resolutions:
                level_blocks.append(AttentionBlock(out_channels))
            self.down_blocks.append(SequentialWithArgs(*level_blocks))
            channels = out_channels
            if i != num_resolutions - 1:
                downsample_layer = ConvBlock(channels, channels, stride=2)
                self.downsample_blocks.append(SequentialWithoutArgs(downsample_layer))

        # --- Middle Block ---
        middle_block_list = []
        middle_block_list.append(ResBlock(channels, channels, time_emb_dim, time_emb_dim if num_classes is not None else None, dropout))
        middle_block_list.append(AttentionBlock(channels))
        middle_block_list.append(ResBlock(channels, channels, time_emb_dim, time_emb_dim if num_classes is not None else None, dropout))
        self.middle_block = SequentialWithArgs(*middle_block_list)

        # --- Upsampling Path ---

        # ***** BUILD UPSAMPLE BLOCKS FIRST *****
        self.upsample_blocks = nn.ModuleList() # Contains SequentialWithoutArgs for upsample layers
        # Need len(model_channels) - 1 upsample blocks
        # These connect the output of one up-level to the input of the next (spatially larger) level's blocks
        upsample_input_channels = model_channels[-1] # Input to first upsample is bottleneck channel count
        for i in range(len(model_channels) - 1):
             # The input channels for the upsample block correspond to the output channels
             # of the block sequence *before* it in the upsampling path.
             level_idx_before_upsample = len(model_channels) - 1 - i # 3, 2, 1
             input_ch_upsample = model_channels[level_idx_before_upsample]

             upsample_layer = nn.Sequential(
                 nn.Upsample(scale_factor=2, mode='nearest'),
                 ConvBlock(input_ch_upsample, input_ch_upsample) # Upsample+ConvBlock maintains channel count
             )
             self.upsample_blocks.append(SequentialWithoutArgs(upsample_layer))
             # The *output* channel count of this upsample block is still input_ch_upsample
             # which becomes part of the input (after concat) to the next level's ResBlocks.

        # ***** BUILD MAIN UP BLOCKS (ResNet + Attention) *****
        self.up_blocks = nn.ModuleList() # Contains SequentialWithArgs for each level
        channels = model_channels[-1] # Start channel count at the bottleneck

        for i in range(num_resolutions): # i = 0, 1, 2, 3 (corresponds to upsample levels)
            level = num_resolutions - 1 - i # Corresponding index in model_channels (3, 2, 1, 0)
            out_channels_stage = model_channels[level] # Target output channels for ResBlocks at this level (256, 128, 64, 32)

            # Determine input channels for the first block in this up level
            # It's current_channels (output from previous up level or middle) + skip_connection_channels
            skip_ch_count = model_channels[level]
            input_channels_first_block = channels + skip_ch_count

            level_blocks = [] # Modules for the current level (ResBlocks + Attention)
            res_input_ch = input_channels_first_block # Input for the first ResBlock in the level
            for j in range(self.num_res_blocks):
                 level_blocks.append(ResBlock(res_input_ch, out_channels_stage, time_emb_dim, time_emb_dim if num_classes is not None else None, dropout))
                 res_input_ch = out_channels_stage # Subsequent ResBlocks take the target channels

            spatial_size = LATENT_SPATIAL_SIZE // (2**(num_resolutions - 1 - i)) # Use current level's spatial size
            if spatial_size in attention_resolutions:
                 # Attention takes the *output* channels of the ResBlocks at this level
                 level_blocks.append(AttentionBlock(out_channels_stage))

            # Group level blocks in SequentialWithArgs
            self.up_blocks.append(SequentialWithArgs(*level_blocks))

            # The output channels of this level is 'out_channels_stage'
            channels = out_channels_stage # Update channels for the next iteration's input calculation


        # --- Final Output Layers ---
        self.out = nn.Sequential(
            nn.GroupNorm(32, channels), # channels should be model_channels[0] here
            nn.GELU(),
            nn.Conv2d(channels, 8, kernel_size=3, padding=1)
        )

    # --- forward method remains the same as the 'CORRECT' version in the previous answer ---
    def forward(self, x, t, y=None):
        time_emb = self.time_embed(t)
        class_emb = None
        if self.class_embed is not None and y is not None:
            class_emb = self.class_embed(y)

        hs = [] # Store outputs of down_blocks (before downsampling)

        # Initial input conv block
        h = self.initial_conv(x)

        # Downsampling path
        for i in range(len(self.down_blocks)):
            h = self.down_blocks[i](h, time_emb, class_emb) # Call SequentialWithArgs
            hs.append(h)
            if i < len(self.downsample_blocks):
                h = self.downsample_blocks[i](h) # Call SequentialWithoutArgs

        # Middle block
        h = self.middle_block(h, time_emb, class_emb) # Call SequentialWithArgs

        # Upsampling path
        skip_hs = hs[::-1]
        for i in range(len(self.up_blocks)): # i = 0, 1, 2, 3
            # Apply upsample layer *before* processing the blocks for this level (if not the very first up level)
            if i != 0: # Apply upsample before up_blocks[1], up_blocks[2], up_blocks[3]
                # Use the pre-built upsample block: upsample_blocks index i-1
                h = self.upsample_blocks[i-1](h) # <--- Call SequentialWithoutArgs for upsample

            # Get skip connection for this level
            skip_h = skip_hs[i]

            # Concatenate h (output of previous upsample or middle) with skip_h
            h = torch.cat([h, skip_h], dim=1)

            # Process level blocks with the concatenated input
            h = self.up_blocks[i](h, time_emb, class_emb) # Call SequentialWithArgs

        # Final output layers
        return self.out(h)
