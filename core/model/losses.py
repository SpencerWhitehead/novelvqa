import torch
import torch.nn.functional as F
from torch import nn

from math import sqrt
import numpy as np


class ContrastProjection(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.linear1 = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear2 = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

    def forward(self, tokens):
        return self.linear2(F.relu(self.linear1(tokens)))


class Losses:
    def __init__(self, __C):
        self.__C = __C
        self.maskval = -1e9
        if __C.USE_GROUNDING:
            self._point_loss = nn.CrossEntropyLoss().cuda()
        else:
            self._point_loss = None

        if __C.SKILL_CONT_LOSS:
            self._skill_contrast_proj = ContrastProjection(__C).cuda()
            self._skill_contrast_loss = nn.CrossEntropyLoss().cuda()
        else:
            self._skill_contrast_proj = None
            self._skill_contrast_loss = None

        self._skill_pool_method = __C.SKILL_POOL

        self._skill_temp = __C.SK_TEMP

        self._point_temp = __C.PT_TEMP

    def get_pointing_scores(self, tgt, refs, ref_masks, point_mask_tok):
        # tgt: size: batch x sent_len x 512
        # refs[i]: size: batch x sent_len x 512
        # ref_masks[i]: batch x sent_len; indicates the locations where there is padding (1 if the index is padding, 0 otherwise)
        # point_mask_tok: batch x 1; vector indicating where in the target sequence is the masked token used for pointing

        batch_size, num_toks, tok_dim = tgt.size()
        n_refs = len(refs)

        row_id = torch.from_numpy(np.array(range(batch_size)))
        masked_tok = tgt[row_id, point_mask_tok.squeeze(1)]  # batch_size x tok_dim

        all_ref_hiddens = torch.cat(refs, dim=1)
        all_ref_masks = torch.cat(ref_masks, dim=-1)

        scores = torch.zeros(batch_size, num_toks * n_refs, dtype=tgt.dtype, device=tgt.device)

        for i in range(batch_size):
            scores[i, :] = torch.matmul(masked_tok[i], all_ref_hiddens[i].t()) / sqrt(tok_dim)

        logits = scores.masked_fill(all_ref_masks, self.maskval)  # mask out padding

        return logits, F.softmax(logits, dim=-1)

    def pointing_loss(self, tgt, refs, ref_masks, point_mask_tok, pos):
        logits, _ = self.get_pointing_scores(tgt, refs, ref_masks, point_mask_tok)
        point_loss_ = self._point_loss(logits, pos.squeeze(1))
        return point_loss_

    def skill_contrast_loss(self, tgt_tokens, tgt_mask, all_ref_tokens, ref_masks, ref_labels):
        # tgt_tokens: batch x 1 x dim  OR  batch x # tokens x dim (if pool_method is given)
        # all_ref_tokens: [batch x 1 x dim  OR  batch x # tokens x dim] x # refs

        if self._skill_pool_method in {'mean', 'max'}:
            tgt_tokens.masked_fill_(tgt_mask.unsqueeze(2), 0.)

            if self._skill_pool_method == 'mean':
                tgt_tokens = torch.mean(tgt_tokens, dim=1, keepdim=True)
            elif self._skill_pool_method == 'max':
                tgt_tokens = torch.max(tgt_tokens, dim=1, keepdim=True)

            masked_ref_tokens = []

            for rt, rm in zip(all_ref_tokens, ref_masks):

                rt.masked_fill_(rm.unsqueeze(2), 0.)

                if self._skill_pool_method == 'mean':
                    rt = torch.mean(rt, dim=1, keepdim=True)
                elif self._skill_pool_method == 'max':
                    rt = torch.max(rt, dim=1, keepdim=True)
                masked_ref_tokens.append(rt)

            all_ref_tokens = torch.cat(masked_ref_tokens, dim=1)  # batch x # refs x D
        else:
            all_ref_tokens = torch.cat(all_ref_tokens, dim=1)  # batch x # refs x D

        tgt_tokens = self._skill_contrast_proj(tgt_tokens)
        all_ref_tokens = self._skill_contrast_proj(all_ref_tokens)

        norm_tgt_cls = nn.functional.normalize(tgt_tokens, p=2, dim=-1)
        norm_all_ref_cls = nn.functional.normalize(all_ref_tokens, p=2, dim=-1)

        sims_ = torch.bmm(norm_all_ref_cls, norm_tgt_cls.permute(0, 2, 1)).squeeze(2)

        sims_ = torch.div(sims_, self._skill_temp)

        return self._skill_contrast_loss(sims_, ref_labels.squeeze(-1))
