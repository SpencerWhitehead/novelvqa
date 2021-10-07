from core.data.ans_punct import prep_ans

from core.data.save_glove_embeds import StoredEmbeds
import numpy as np
import random, re, json
from torch.utils.data._utils.collate import default_collate


try:
    import en_vectors_web_lg
except ImportError:
    import spacy


def shuffle_list(ans_list):
    random.shuffle(ans_list)


def save_json(obj, fname):
    with open(fname, 'w') as f:
        json.dump(obj, f)


def load_json(fname):
    with open(fname, 'r') as f:
        data_ = json.load(f)
    return data_

# ------------------------------
# ---- Initialization Utils ----
# ------------------------------

def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path


def img_feat_load(path_list):
    iid_to_feat = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        img_feat = np.load(path)
        img_feat_x = img_feat['x'].transpose((1, 0))
        iid_to_feat[iid] = img_feat_x
        print('\rPre-Loading: [{} | {}] '.format(ix, path_list.__len__()), end='          ')

    return iid_to_feat


def ques_load(ques_list):
    qid_to_ques = {}

    for ques in ques_list:
        qid = str(ques['question_id'])
        qid_to_ques[qid] = ques

    return qid_to_ques


def get_words(question_str):
    return re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        question_str.lower()
    ).replace('-', ' ').replace('/', ' ').split()


def tokenize(stat_ques_list, use_glove, save_embeds=False):
    # This function basically requires use_glove to be true in order to work correctly.
    # Otherwise, the indicies in token_to_ix don't match the actual embedding matrix.

    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
        '[MASK]': 2,
        '[CLS]': 3
    }

    spacy_tool = None
    pretrained_emb = []
    stored_embeds = StoredEmbeds(embed_fname='./ckpts/glove_embeds.pkl')
    if use_glove:
        try:
            spacy_tool = en_vectors_web_lg.load()
        except NameError:
            try:
                spacy_tool = spacy.load('en_vectors_web_lg')
            except OSError:
                if not stored_embeds.has_embeds():
                    raise ValueError('Spacy could not be loaded and no stored glove embeddings were found.')
                return stored_embeds.get_embeds()

        known_vec = spacy_tool('the').vector
        mu = 0.
        sigma = np.sqrt(1. / known_vec.shape[0])

        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(
            sigma * np.random.randn(*known_vec.shape).astype(dtype=known_vec.dtype) + mu
        )  # Embedding for [MASK]
        pretrained_emb.append(
            sigma * np.random.randn(*known_vec.shape).astype(dtype=known_vec.dtype) + mu
        )  # Embedding for [CLS]

    for ques in stat_ques_list:
        words = get_words(ques['question'])

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    if save_embeds:
        # Embeddings will not be overwritten if file already exists.
        stored_embeds.set_embeds(token_to_ix, pretrained_emb)
        stored_embeds.save()

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


# def ans_stat(stat_ans_list, ans_freq):
#     ans_to_ix = {}
#     ix_to_ans = {}
#     ans_freq_dict = {}
#
#     for ans in stat_ans_list:
#         ans_proc = prep_ans(ans['multiple_choice_answer'])
#         if ans_proc not in ans_freq_dict:
#             ans_freq_dict[ans_proc] = 1
#         else:
#             ans_freq_dict[ans_proc] += 1
#
#     ans_freq_filter = ans_freq_dict.copy()
#     for ans in ans_freq_dict:
#         if ans_freq_dict[ans] <= ans_freq:
#             ans_freq_filter.pop(ans)
#
#     for ans in ans_freq_filter:
#         ix_to_ans[ans_to_ix.__len__()] = ans
#         ans_to_ix[ans] = ans_to_ix.__len__()
#
#     return ans_to_ix, ix_to_ans

def ans_stat(json_file):
    ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

    return ans_to_ix, ix_to_ans


# ------------------------------------
# ---- Real-Time Processing Utils ----
# ------------------------------------

def proc_img_feat(img_feat, img_feat_pad_size):
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat


def proc_ques(ques, token_to_ix, max_token, add_cls=False):
    if not add_cls:
        ques_ix = np.zeros(max_token, np.int64)
        start_ix = 0
        max_len = max_token
    else:
        ques_ix = np.zeros(max_token + 1, np.int64)
        ques_ix[0] = token_to_ix['[CLS]']
        start_ix = 1
        max_len = max_token + 1

    words = get_words(ques['question'])

    for ix, word in enumerate(words, start=start_ix):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_len:
            break

    return ques_ix


def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.


def proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}

    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

    return ans_score


def refset_collate(batch):

    tgt, refs, label, pos, qid_data = zip(*batch)

    tgt, label, pos = default_collate(tgt), default_collate(label), default_collate(pos)

    refs = list(refs)
    n_refs = len(refs[0])  # number of reference examples

    batched_refs = []
    for i in range(n_refs):
        ref_i_all = [per_row[i] for per_row in refs]
        ref_i_batched = default_collate(ref_i_all)
        batched_refs.append(ref_i_batched)


    return tgt, batched_refs, label, pos, qid_data


def refset_tocuda(refset_data):
    tgt, batched_refs, label, pos, qid_data = refset_data
    label, pos = label.cuda(), pos.cuda()

    tgt = (tgt[0].cuda(),tgt[1].cuda(), tgt[2].cuda())

    if all(len(x) for x in batched_refs):
        batched_refs = [(x[0].cuda(), x[1].cuda(), x[2].cuda()) for x in batched_refs]

    return tgt, batched_refs, label, pos, qid_data


def refset_point_refset_index(question_list, max_token, novel_indices=None, aug_factor=1):
    # This assumes that each concept only appears once in the question.
    # If this is a bad assumption, then we need to iterate over question['concepts']

    n_questions = len(question_list)
    is_novel = [False for _ in range(n_questions)]
    if novel_indices:
        for x in novel_indices:
            is_novel[x] = True

    rs_idx = []
    count_novel = 0
    for qidx, question in enumerate(question_list):
        if question.get('refsets', None):
            for c, crefs in question['refsets'].items():
                has_refs = True
                for dkey, vals in crefs.items():
                    if not len(vals['index']) or not len(vals['question_id']):
                        has_refs = False
                        break

                # Assumes each concepts appears once.
                if get_concept_position(question, c) < max_token and has_refs:
                    rs_idx.append((qidx, c))

                    if is_novel[qidx]:
                        for _ in range(aug_factor-1):
                            rs_idx.append((qidx, c))
                            count_novel += 1

    print('Added {x} number of novel questions for the refset'.format(x=count_novel))

    return rs_idx


def prune_refsets(question_list, refset_sizes, max_token):
    for question in question_list:
        if question.get('refsets', None):
            for c, crefs in question['refsets'].items():
                for dkey, _ in refset_sizes:
                    for i, idx in reversed(list(enumerate(crefs[dkey]['index']))):
                        if len(get_words(question_list[idx]['question'])) > max_token:
                            crefs[dkey]['index'].pop(i)
                            crefs[dkey]['question_id'].pop(i)

    return question_list


def get_concept_position(question, concept):
    # This assumes that concepts are only 1 word. Also, as in refset_index(), we assume each concept appears once.
    return question['concepts'].get(concept, [[-1]])[0][0]


def do_token_masking(token_id, token_to_ix, mask_mode):
    # target = do what we do now
    # bert = do what BERT does
    # even = 50 / 50 Mask or keep same
    # https://github.com/google-research/bert/blob/0fce551b55caabcfba52c61e18f34b541aef186a/create_pretraining_data.py#L342
    masked_token_id = None
    if mask_mode == 'target':
        masked_token_id = token_to_ix['[MASK]']
    elif mask_mode == 'bert':
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token_id = token_to_ix['[MASK]']
        else:
            # 10% of the time, keep original
            if random.random() <= 0.5:
                masked_token_id = token_id
            # 10% of the time, replace with random word
            else:
                masked_token_id = random.randint(4, len(token_to_ix) - 1)  # start at 4 to account for PAD, UNK, MASK, CLS
    elif mask_mode == 'even':
        if random.random() <= 0.5:
            masked_token_id = token_to_ix['[MASK]']
        else:
            masked_token_id = token_id
    elif mask_mode is None or mask_mode.lower() == 'none':
        masked_token_id = token_id
    else:
        raise ValueError('mask_mode must be in [target, bert, even, none/None]')

    return masked_token_id


def filter_concept_skill(ques_list, ans_list, concept, skill):
    N, N_ans = len(ques_list), len(ans_list)
    assert N == N_ans

    novel_ques_ids, novel_indices = get_novel_ids(ques_list, concept, skill)

    count = 0
    for id in reversed(novel_indices): # going back to front, delete novel idx
        del ques_list[id]
        del ans_list[id]
        count += 1

    print('Removed {x} number of novel questions from the current split'.format(x=count))
    print('New dataset size is {x}'.format(x=len(ques_list)))


def get_novel_ids(ques_list, concept, skill):
    novel_ids, novel_indices = [], []
    if not concept: return novel_ids, novel_indices

    if isinstance(concept, str):
        concept = concept.split(',')

    concept_set = set(concept)

    N = len(ques_list)

    for i in range(N):
        ques = ques_list[i]

        if 'all_concepts' not in ques:
            curr_concepts = set(ques['concepts'])
        else:
            curr_concepts = set(ques['all_concepts'])

        found_concept = bool(len(concept_set & curr_concepts))

        if not found_concept:
            continue

        if (skill is None or skill.lower() == 'none') or ques['skill'] == skill:
            # Found a match, add question id
            novel_ids.append(ques['question_id'])
            novel_indices.append(i)

    print('Found {x} number of novel question ids'.format(x= len(novel_ids)))
    return novel_ids, novel_indices


def sample_references(question, concept, reftype_key, n_samples=1):
    return random.sample(question['refsets'][concept][reftype_key]['index'], k=n_samples)


def sample_refset(question, concept, refset_sizes):
    sampled_rs = []
    for dkey, n_samples in refset_sizes:
        sampled_rs.append(sample_references(question, concept, dkey, n_samples))
    return sampled_rs


def build_skill_references(question_list):
    skill_refs = []
    for i, ques in enumerate(question_list):
        if ques.get('skill_refset', None):
            if len(ques['skill_refset']['pos']) and len(ques['skill_refset']['neg']) > 1:
                skill_refs.append((i, 'none'))
    return skill_refs


def sample_contrasting_skills(question, n_pos_samples, n_neg_samples):
    pos_samples_ = random.sample(question['skill_refset']['pos'], k=n_pos_samples)
    neg_samples_ = random.sample(question['skill_refset']['neg'], k=n_neg_samples)
    return pos_samples_, neg_samples_
