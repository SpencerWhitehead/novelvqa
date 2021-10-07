from core.data.load_data import DataSet, RefPointDataSet, SkillContrastDataSet
from core.model.PointNet import PointNet
from core.model.optim import get_optim, adjust_lr
from core.model.losses import Losses
from core.data.data_utils import shuffle_list, refset_collate, refset_tocuda
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import random


class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading training set ........')


        if __C.CONCEPT is not None or __C.SKILL is not None: # take out novel concept/skill from training
            setattr(__C, 'NOVEL', 'remove')
        else:
            setattr(__C, 'NOVEL', 'get_ids')

        self.dataset = DataSet(__C)

        if self.__C.USE_GROUNDING:
            __C_ref = copy.deepcopy(__C)
            setattr(__C_ref, 'NOVEL', 'augment')
            self.refdataset = RefPointDataSet(__C_ref)
        else:
            self.refdataset = None

        if self.__C.SKILL_CONT_LOSS:
            __C_sk_ref = copy.deepcopy(__C)
            setattr(__C_sk_ref, 'NOVEL', 'augment')
            self.sk_contrast_dataset = SkillContrastDataSet(__C_sk_ref)
        else:
            self.sk_contrast_dataset = None

        self.dataset_eval = None
        if __C.EVAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(__C)
            setattr(__C_eval, 'RUN_MODE', 'val')
            setattr(__C_eval, 'NOVEL', 'get_ids')  # for validation, we just need the ids to compute accuracy

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(__C_eval)

    def train(self, dataset, refdataset=None, sk_contdataset=None, dataset_eval=None):
        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        loss_fns = Losses(self.__C)

        # Define the model
        net = PointNet(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size)

        print(net)

        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        # Define the binary cross entropy loss
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Load checkpoint if resume training
        if self.__C.RESUME:
            print('========== Resume training ==========')

            if self.__C.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.__C.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, net, data_size)
            start_epoch = 0

        loss_sum = 0

        # Define multi-thread dataloader
        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )

        if self.__C.USE_GROUNDING:
            refsetloader = Data.DataLoader(
                refdataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True,
                collate_fn=refset_collate
            )

            refsetloader_iter = iter(refsetloader)

        if self.__C.SKILL_CONT_LOSS:
            sk_contloader = Data.DataLoader(
                sk_contdataset,
                batch_size=self.__C.BATCH_SIZE // 4,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True,
                collate_fn=refset_collate
            )
            sk_contloader_iter = iter(sk_contloader)

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            # Externally shuffle
            if self.__C.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):

                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]

                    output = net(sub_img_feat_iter, sub_ques_ix_iter)
                    pred = output[0]

                    loss = loss_fn(pred, sub_ans_iter)

                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because of the Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.__C.GRAD_ACCU_STEPS
                    loss.backward()
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']

                        print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate
                        ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                optim.step()

                if self.__C.USE_GROUNDING and random.random() <= self.__C.GROUNDING_PROB:
                    optim.zero_grad()

                    try:
                        point_batch = next(refsetloader_iter)
                    except StopIteration:
                        refsetloader_iter = iter(refsetloader)
                        point_batch = next(refsetloader_iter)

                    target, refs, mask_tok_pos, point_positions, qid_data = refset_tocuda(point_batch)

                    # -------------- Forward pass: target and refs ---------------- #
                    output = net(target[0], target[1])
                    target_vqa_output, target_hiddens = output[0], output[1]

                    # -------------- Compute loss of pointing  -------------- #
                    refs_vqa_output, refs_hiddens, refs_masks = [], [], []
                    for i in range(len(refs)):
                        ref_output = net(refs[i][0], refs[i][1])
                        r_output, r_hidden, r_hidden_mask = ref_output[0], ref_output[1], ref_output[2]
                        refs_vqa_output.append(r_output)
                        refs_hiddens.append(r_hidden)
                        refs_masks.append(r_hidden_mask.squeeze(2).squeeze(1))

                    loss_pointing = loss_fns.pointing_loss(
                        target_hiddens, refs_hiddens, refs_masks, mask_tok_pos, point_positions
                    )

                    if self.__C.SKILL_CONT_LOSS:
                        try:
                            sk_cont_batch = next(sk_contloader_iter)
                        except StopIteration:
                            sk_contloader_iter = iter(sk_contloader)
                            sk_cont_batch = next(sk_contloader_iter)

                        target, refs, _, point_positions, _ = refset_tocuda(sk_cont_batch)

                        output = net(target[0], target[1])

                        if self.__C.SKILL_POOL == 'cls':
                            target_tokens = output[-1][1]
                            target_mask = None
                        else:
                            target_tokens = output[1]
                            target_mask = output[2].squeeze(2).squeeze(1)


                        # -------------- Compute skill loss  -------------- #

                        refs_tokens = []
                        refs_masks = []
                        for i in range(len(refs)):
                            ref_output = net(refs[i][0], refs[i][1])

                            if self.__C.SKILL_POOL == 'cls':
                                r_token = ref_output[-1][1]
                                r_token_mask = None
                            else:
                                r_token = ref_output[1]
                                r_token_mask = ref_output[2].squeeze(2).squeeze(1)

                            refs_tokens.append(r_token)
                            refs_masks.append(r_token_mask)

                        loss_sk_cont = loss_fns.skill_contrast_loss(
                            target_tokens,
                            target_mask,
                            refs_tokens,
                            refs_masks,
                            point_positions,
                        )

                        loss_pointing += loss_sk_cont

                    loss_pointing.backward()
                    optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                # '/epoch' + str(epoch_finish) +
                '/last_epoch.pkl'
            )

            # Logging
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                with torch.no_grad():
                    self.eval(
                        dataset_eval,
                        state_dict=net.state_dict(),
                        valid=True
                    )

            loss_sum = 0

    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        if self.__C.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__C.CKPT_PATH
        else:
            path = self.__C.CKPTS_PATH + \
                   'ckpt_' + self.__C.CKPT_VERSION + \
                   '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb
        novel_ques_ids = dataset.novel_ques_ids

        # Define the model
        net = PointNet(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size)

        net.cuda()
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            output = net(img_feat_iter, ques_ix_iter)
            pred = output[0]
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # Save the answer index
            if pred_argmax.shape[0] != self.__C.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)

            # Save the whole prediction vector
            if self.__C.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.__C.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.__C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

        print('')
        ans_ix_list = np.array(ans_ix_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])
        } for qix in range(qid_list.__len__())]

        # Write the results to result file
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.VERSION + \
                    '.json'

        else:
            if self.__C.CKPT_PATH is not None:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.__C.TEST_SAVE_PRED:

            if self.__C.CKPT_PATH is not None:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            } for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)

        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.__C.QUESTION_PATH['val']
            ans_file_path = self.__C.ANSWER_PATH['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            vqaEval.evaluate()

            # print accuracies
            print("\n")
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")

            if type(novel_ques_ids) is list and len(novel_ques_ids):
                # evaluate results on novel subset

                vqaEval.evaluate(novel_ques_ids)

                # print accuracies
                print("\n")
                print("Novel Subset Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
                print("Per Answer Type Accuracy is the following:")
                for ansType in vqaEval.accuracy['perAnswerType']:
                    print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
                print("\n")

            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+'
                )

            else:
                print('Write to log file: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+'
                )

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n\n")
            logfile.close()

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)

            refdata = None
            if self.__C.USE_GROUNDING:
                refdata = self.refdataset

            skcontdata = None
            if self.__C.SKILL_CONT_LOSS:
                skcontdata = self.sk_contrast_dataset

            self.train(
                self.dataset,
                refdataset=refdata,
                sk_contdataset=skcontdata,
                dataset_eval=self.dataset_eval
            )

        elif run_mode == 'val':
            with torch.no_grad():
                self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            with torch.no_grad():
                self.eval(self.dataset)

        else:
            exit(-1)

    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')




