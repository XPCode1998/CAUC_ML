class TDM_Net(nn.Module):
    def __init__(self, input_dim, diff_channels, diff_steps, diff_layers, dilation, c_step, c_cond, unconditional):
        super(TDM_Net, self).__init__()

        # 输入投影
        self.input_projection = Conv1d(input_dim, diff_channels, 1)

        # 扩散步嵌入
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=c_step, projection_dim=c_step)

        if not unconditional:
            self.spectrogram_upsampler = SpectrogramUpsampler(c_cond)
        else:
            self.spectrogram_upsampler = None

        # 残差层
        self.residual_layers = nn.ModuleList([
            TDM_Residual_Block(diff_channels, c_step, c_cond, 2 ** (i % dilation), uncond=unconditional)
            for i in range(diff_layers)
        ])

        # 跳跃投影
        self.skip_projection = Conv1d(diff_channels, diff_channels, 1)
        self.output_projection = Conv1d(diff_channels, input_dim, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input_data, spectrogram=None):
        x, cond, data_stamp, mask, t = input_data

        assert (spectrogram is None and self.spectrogram_upsampler is None) or (
                spectrogram is not None and self.spectrogram_upsampler is not None)

        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(t)
        if self.spectrogram_upsampler:
            spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)

        return x
