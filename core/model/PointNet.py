from core.model.net_utils import LayerNorm
from core.model.mca import SA, AttFlat

import torch.nn as nn
import torch

import math


class MCA_Unified(nn.Module):
    def __init__(self, __C, answer_size=0):
        super(MCA_Unified, self).__init__()

        # add tokens for answer set
        self.cls_token = nn.Parameter(
            torch.zeros(1, __C.HIDDEN_SIZE).normal_(mean=0, std=math.sqrt(1 / __C.HIDDEN_SIZE)))

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask, layer_id=0):
        batch_size = x.size(0)

        cls_token_ = self.cls_token.expand(batch_size, *self.cls_token.size())

        cls_mask = torch.zeros((batch_size, 1, 1, 1), dtype=x_mask.dtype, device=x_mask.device)

        chunks = [1, x.size(1), y.size(1)]  # cls, x.size(1), y.size(1)
        t_size = x.size(1)
        im_size = y.size(1)
        combo = torch.cat([cls_token_, x, y], dim=1)
        combo_mask = torch.cat([cls_mask, x_mask, y_mask], dim=-1)
        combo_mask[:, :, 1:1 + t_size, -im_size::] = True  # no text->img direct attn

        attmap_list = []
        hidden_text_list = []
        for enc in self.enc_list:
            combo, attmap = enc(combo, combo_mask)
            attmap_list.append(attmap.unsqueeze(1))  # make a new dimension for the layer dimension concatenation

            c, x, y = torch.split(combo, chunks, dim=1)
            hidden_text_list.insert(0, x)   # last layer first, then second last layer

        text_hiddens = hidden_text_list[layer_id]
        attmap_list = torch.cat(attmap_list, 1) # batch x layer x head x tokid x tokid
        others = attmap_list, c, text_hiddens   # let's get class token output as well
        return x, y, others


# -------------------------
# ---- Main MCAN Model ----
# -------------------------
class PointNet(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(PointNet, self).__init__()

        self.GROUND_LAYER = getattr(__C, 'GROUND_LAYER', 0)

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )
        # Loading the GloVe embedding weights
        if __C.USE_GLOVE and pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )
        self.backbone = MCA_Unified(__C, answer_size)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)


        if __C.USE_POINT_PROJ:
            self.point_proj = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        else:
            self.point_proj = lambda x: x

        self.proj_norm = LayerNorm(__C.HIDDEN_SIZE)
        self.linear_proj = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.classifier = nn.Linear(__C.HIDDEN_SIZE, answer_size)

    def forward(self, img_feat, ques_ix, **kwargs):
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat, others = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
            layer_id=self.GROUND_LAYER
        )

        c, lang_hiddens = others[1], others[2]
        if self.training:
            lang_hiddens_out = self.point_proj(lang_hiddens)
        else:
            lang_hiddens_out = lang_hiddens

        # close to mcan's output layer
        proj_feat = torch.sigmoid(self.classifier(self.proj_norm(self.linear_proj(c))))

        proj_feat = torch.squeeze(proj_feat)

        ret_others = [*others] + [img_feat, img_feat_mask]

        return proj_feat, lang_hiddens_out, lang_feat_mask, ret_others

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
