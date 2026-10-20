import numpy as np
from sklearn.metrics import *



class printer:
    def info(self, message, format=None):
        print(message)
    def warning(self, message, format=None):
        print(message)

class nonprinter:
    def info(self, message, format=None):
        pass
    def warning(self, message, format=None):
        pass


def assign_weights(length, first_weight = 0.5):
    if length < 1:
        raise ValueError("列表长度至少为 1")
    if length == 1:
        return [1.0]  # 如果列表长度为 1，直接返回权重为 1

    # 剩余权重
    remaining_weight = 1 - first_weight

    # 其他值的权重
    other_weight = remaining_weight / (length - 1)

    # 生成权重列表
    weights = [first_weight] + [other_weight] * (length - 1)
    return weights

class Evaluation():
    def __init__(self, logger=printer(), metrics='all'):
        if logger is None: logger=nonprinter()
        logger.info('evaluation', '%(asctime)s ----%(message)s---')
        self.logger = logger
        if metrics == 'all':
            self.metrics = ['acc', 'auc', 'auroc', 'auprc', 'precision', 'f1', 'recall', 'MCC']
        else:
            self.metrics = metrics
        self.weigths = assign_weights(len(self.metrics))

    
    def cacu1(self, name, predict_proba, groundtruth):

        predict = np.argmax(np.array(predict_proba), axis=1)
        confusion = confusion_matrix(groundtruth, predict)
        self.logger.info(f'confusion:{confusion}', 0)

        not_capable = False if (confusion[0][0] and confusion[1][1]) else True
        metirc_dict = {} 
        if 'acc' in self.metrics:
            metirc_dict['acc'] = accuracy_score(groundtruth, predict)
        if 'auc' in self.metrics:
            metirc_dict['auc'] = roc_auc_score(groundtruth, predict_proba[:, 1])
        if 'precision' in self.metrics:
            metirc_dict['precision'] = 0 if not_capable else precision_score(groundtruth, predict)
        if 'f1' in self.metrics:
            metirc_dict['f1'] = 0 if not_capable else f1_score(groundtruth, predict, pos_label= 1, average='binary')
        if 'recall' in self.metrics:
            metirc_dict['recall'] = 0 if not_capable else recall_score(groundtruth, predict)
        # self.logger.info([(i, metirc_dict[i]) for i in metirc_dict.keys()], 0)
        self.logger.info(metirc_dict, 0)
        return metirc_dict
    
    def cacu2(self, name, predict_proba, groundtruth):
        y_score = np.array(predict)[:,1]
        predict = np.argmax(np.array(predict), axis=1)
        confusion = confusion_matrix(groundtruth, predict)
        self.logger.info(f'confusion:{confusion}', 0)

        # not_capable = False if (confusion[0][0] and confusion[1][1]) else True
        metirc_dict = {} 
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 
        if 'acc' in self.metrics:
            metirc_dict['acc'] = round(float(TN + TP) / (TN + FP + FN + TP), 5)
        if 'auc' in self.metrics:
            metirc_dict['auc'] = round(roc_auc_score(groundtruth, predict_proba[:, 1]), 5)
        if 'auroc' in self.metrics:  # 和auc相比，auroc使用的y_score指的是模型输出的预测概率值
            fpr, tpr, _ = roc_curve(groundtruth, y_score)
            metirc_dict['auroc'] = round(auc(fpr, tpr), 5)
        if 'auprc' in self.metrics:  # auprc使用的y_score指的是模型输出的预测概率值
            # 这样做pcr预测的时候，评价指标就很全面了，而且对于结果分析和对比的时候，我们就可以展开讲讲了，比以前设计的内容更多
            # 比如：我们的AUPRC高于ResNet-50，就说明我们的模型更适合于这种类别不平衡（即：正例很少的数据集），更加临床实际情况，同时，也从侧面证明了我们的模型没有过拟合。
            precision, recall, _ = precision_recall_curve(groundtruth, y_score, pos_label=1)
            metirc_dict['auprc'] = round(auc(recall, precision), 5)
        if 'precision' in self.metrics:
            metirc_dict['precision'] = 0 if (FP + TP)==0 else round(float(TP) / (FP + TP), 5)
        if 'f1' in self.metrics:
            metirc_dict['f1'] = 0 if (2 * TP + FP + FN) == 0 else round(float(2 * TP) / (2 * TP + FP + FN), 5)
        if 'recall' in self.metrics:
            metirc_dict['recall'] = 0 if (FN + TP)==0 else round(float(TP) / (FN + TP), 5)
        if 'MCC' in self.metrics:
            denominator_square = (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)
            metirc_dict['MCC'] = -1 if denominator_square==0 else round(float(TP * TN - FP * FN)/np.sqrt(denominator_square), 5)
        self.logger.info(metirc_dict, 0)
        return metirc_dict

    def view(self, name, predict, groundtruth, print_probability):
        if print_probability:
            self.logger.info(f'probability:{[(n, p) for n, p in zip(name, predict)]}', 0)
        metirc_dict = self.cacu2(name, predict, groundtruth)
        # return metirc_dict[metirc]
        return sum(metirc*weight for metirc, weight in zip(metirc_dict.values(), self.weigths))   

