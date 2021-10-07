from core.data.load_data import DataSet
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, copy


class Execution:
    def __init__(self, __C):

        self.__C = __C
        __C_eval = copy.deepcopy(__C)
        setattr(__C_eval, 'RUN_MODE', 'val')
        print('Loading validation set for per-epoch evaluation ........')
        self.dataset_eval = DataSet(__C_eval)

    def run(self, run_mode):
        if run_mode == 'valNovel':
            self.eval(self.dataset_eval)
        else:
            exit(-1)

    # Evaluation
    def eval(self, dataset):

        print(self.__C.RESULT_EVAL_FILE)

        # Load parameters
        if self.__C.RESULT_EVAL_FILE is None:
            exit(-1)

        if not os.path.isfile(self.__C.RESULT_EVAL_FILE):
            exit(-1)

        result_eval_file = self.__C.RESULT_EVAL_FILE

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
        # print("Per Question Type Accuracy is the following:")
        # for quesType in vqaEval.accuracy['perQuestionType']:
        #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
        # print("\n")
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

        novel_ques_ids = dataset.novel_ques_ids
        if self.__C.NOVEL and novel_ques_ids is not None:
            # evaluate results on novel subset

            vqaEval.evaluate(novel_ques_ids)

            # print accuracies
            print("\n")
            print("Novel Subset Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")
