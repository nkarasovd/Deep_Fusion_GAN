import torch
import torch.nn as nn
from torch import Tensor

from src.discriminator.residual_block import ResidualBlockD


class Discriminator(nn.Module):
    def __init__(self, n_c: int, sentence_embed_dim: int = 256):
        super().__init__()
        self.img_forward = nn.Sequential(
            # [batch_size, 3, h, w]
            nn.Conv2d(3, n_c, kernel_size=3, stride=1, padding=1),
            # [batch_size, 32, h, w]
            ResidualBlockD(n_c * 1, n_c * 2),
            # [batch_size, 64, h // 2, w // 2]
            ResidualBlockD(n_c * 2, n_c * 4),
            # [batch_size, 128, h // 4, w // 4]
            ResidualBlockD(n_c * 4, n_c * 8),
            # [batch_size, 256, h // 8, w // 8]
            ResidualBlockD(n_c * 8, n_c * 16),
            # [batch_size, 256, h // 16, w // 16]
            ResidualBlockD(n_c * 16, n_c * 16),
            # [batch_size, 256, h // 32, w // 32]
            ResidualBlockD(n_c * 16, n_c * 16)
            # [batch_size, 256, h // 64, w // 64]
        )

        in_c_logit = 16 * n_c + sentence_embed_dim
        self.img_sentence_forward = nn.Sequential(
            nn.Conv2d(in_c_logit, n_c * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_c * 2, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def build_embeds(self, image: Tensor) -> Tensor:
        # [batch_size, 3, 256, 256]
        out = self.img_forward(image)
        # [batch_size, 512, 4, 4]

        return out

    def get_logits(self, image_embed: Tensor, sentence_embed: Tensor) -> Tensor:
        # bs = batch_size or batch_size - 1

        # image_embed.shape = [bs, 512, 4, 4]
        # sentence_embed.shape = [bs, sentence_embed_dim]

        sentence_embed = sentence_embed.view(-1, 256, 1, 1)
        # sentence_embed.shape = [bs, sentence_embed_dim, 4, 4]
        sentence_embed = sentence_embed.repeat(1, 1, 4, 4)

        # h_c_code.shape = [bs, 512 + sentence_embed_dim, 4, 4]
        h_c_code = torch.cat((image_embed, sentence_embed), 1)

        # logits.shape = [bs, 1, 1, 1] or [bs - 1, 1, 1, 1]
        logits = self.img_sentence_forward(h_c_code)

        return logits
