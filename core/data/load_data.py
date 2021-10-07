from core.data.data_utils import get_concept_position, prune_refsets
from core.data.data_utils import refset_point_refset_index, sample_refset, do_token_masking
from core.data.data_utils import img_feat_path_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans
from core.data.data_utils import filter_concept_skill, get_novel_ids
from core.data.data_utils import build_skill_references, sample_contrasting_skills
import numpy as np
import random
import glob, json, torch
import torch.utils.data as Data


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C

        # Loading all image paths
        self.img_feat_path_list = []
        img_split_list = __C.SPLIT[__C.RUN_MODE].split('+')

        if self.__C.VQACP:
            img_split_list = ['train', 'val']

        for split in img_split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        # Loading question word list
        stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            if __C.RUN_MODE in ['train', 'vqaAccRegion', 'evalAll']:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

        self.rs_idx = []
        self.qid2bbanns = {}

        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        self.novel_ques_ids = None
        if self.__C.NOVEL == 'remove' and self.__C.RUN_MODE == 'train':
            filter_concept_skill(self.ques_list, self.ans_list, concept=self.__C.CONCEPT, skill=self.__C.SKILL)
        elif self.__C.NOVEL == 'get_ids' and self.__C.RUN_MODE == 'val':
            self.novel_ques_ids, _ = \
                get_novel_ids(self.ques_list, concept=self.__C.CONCEPT, skill=self.__C.SKILL)
        else:
            self.novel_ques_ids, self.novel_indices = \
                get_novel_ids(self.ques_list, concept=self.__C.CONCEPT, skill=self.__C.SKILL)


        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(
            stat_ques_list,
            __C.USE_GLOVE,
            save_embeds=(self.__C.RUN_MODE in {'train'})  # Embeddings will not be overwritten if file already exists.
        )
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers stats
        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!\n')

    def __getitem__(self, idx):
        # For code safety
        img_feat_iter = np.zeros(1)
        img_pos_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train', 'evalAll']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
            img_feat_x = img_feat['x'].transpose((1, 0))

            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(
                ques, self.token_to_ix, self.__C.MAX_TOKEN, add_cls=False
            )

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            img_feat_x = img_feat['x'].transpose((1, 0))

            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN, add_cls=False)

        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)

    def __len__(self):
        return self.data_size


class RefPointDataSet(DataSet):
    def __init__(self, __C):
        super().__init__(__C)
        self.__C = __C

        self.refset_sizes = list(zip(['pos', 'neg1', 'neg2'], [1, 1, 1]))

        self.ques_list = prune_refsets(self.ques_list, self.refset_sizes, self.__C.MAX_TOKEN)

        if not self.rs_idx:
            self.rs_idx = refset_point_refset_index(
                self.ques_list,
                self.__C.MAX_TOKEN,
                self.novel_indices,
                aug_factor=getattr(self.__C, 'NOVEL_AUGMENT', 1)
            )

        print('== Refset Dataset size:\n', len(self.rs_idx))

    def __len__(self):
        return len(self.rs_idx)

    def __getitem__(self, idx):
        target_idx, target_concept = self.rs_idx[idx]
        pos_idx_list, neg1_idx_list, neg2_idx_list = \
            sample_refset(self.ques_list[target_idx], target_concept, self.refset_sizes)
        all_cand_idx = pos_idx_list + neg1_idx_list + neg2_idx_list

        random.shuffle(all_cand_idx)

        # This assumes that there is only one positive example
        data_ref = []
        point_positions = []
        ref_qids = []
        cand_q_len = 0
        for i_cand in all_cand_idx:
            curr_cand = super().__getitem__(i_cand)
            curr_cand_pt_pos = get_concept_position(self.ques_list[i_cand], target_concept)
            if curr_cand_pt_pos > -1:
                curr_cand_pt_pos += cand_q_len
                point_positions.append(curr_cand_pt_pos)

            data_ref.append(curr_cand)
            ref_qids.append(self.ques_list[i_cand]['question_id'])
            cand_q_len += len(curr_cand[2])  # length of current candidate question

        data_target = super().__getitem__(target_idx)
        target_concept_pos = get_concept_position(self.ques_list[target_idx], target_concept)

        assert target_concept_pos != -1

        ori_id = data_target[2][target_concept_pos]
        new_id = do_token_masking(ori_id, self.token_to_ix, self.__C.TGT_MASKING)
        data_target[2][target_concept_pos] = new_id

        cand_labels = [target_concept_pos]

        cand_labels = torch.from_numpy(np.array(cand_labels)).type(torch.LongTensor)
        point_positions = torch.from_numpy(np.array(point_positions)).type(torch.LongTensor)

        qid_data_ = {
            'concept': target_concept,
            'tgt': self.ques_list[target_idx]['question_id'],
            'refs': ref_qids
        }

        return data_target, data_ref, cand_labels, point_positions, qid_data_


class SkillContrastDataSet(DataSet):
    def __init__(self,  __C):
        super().__init__(__C)

        print('Building skill references...')
        self.rs_idx = build_skill_references(self.ques_list)
        print('Training reference set questions with skill references: {}'.format(len(self)))
        self.pretrained_emb = None

    def __len__(self):
        return len(self.rs_idx)

    def __getitem__(self, idx):
        target_idx, target_concept = self.rs_idx[idx]

        pos_idx_list, neg1_idx_list = \
            sample_contrasting_skills(self.ques_list[target_idx], n_pos_samples=1, n_neg_samples=2)

        all_cand_idx = pos_idx_list + neg1_idx_list
        point_positions = [0]

        # This assumes that there is only one positive example
        data_ref = []
        ref_qids = []
        for i_cand in all_cand_idx:
            curr_cand = super().__getitem__(i_cand)
            data_ref.append(curr_cand)
            ref_qids.append(self.ques_list[i_cand]['question_id'])

        data_target = super().__getitem__(target_idx)

        cand_labels = torch.from_numpy(np.array(point_positions)).type(torch.LongTensor)
        point_positions = torch.from_numpy(np.array(point_positions)).type(torch.LongTensor)

        qid_data_ = {
            'concept': target_concept,
            'tgt': self.ques_list[target_idx]['question_id'],
            'refs': ref_qids
        }
        return data_target, data_ref, cand_labels, point_positions, qid_data_
