from unittest import result
import nltk
import jieba
from .base import IMetrics



class Bleu(IMetrics):
    def __init__(self):
        self.model = nltk.translate.bleu_score.sentence_bleu
        self.smoothing_func = nltk.translate.bleu_score.SmoothingFunction().method1
    @property
    def name(self):
        return "bleu-"+str(self.ngram)
    def calc(self, hyps, ref, ngram_list) -> float:
        ref = ' '.join(jieba.cut(ref))
        hyps = ' '.join(jieba.cut(hyps))
        # 获取 n 的值
        result={}
        for ngram in ngram_list:
            n = ngram  # 如果没有传递 n，则使用默认的 ngram
            weights = [1 / n] * n  # 根据传入的 n 计算权重
            print(self.name)
            ref_list=[ref.split()]
            blue_n = self.model(ref_list, hyps.split(), weights=weights, smoothing_function=self.smoothing_func)
            result['blue-'+str(n):blue_n]
        return blue_n


class SelfBleu(IMetrics):
    def __init__(self, ngram=3):
        super(SelfBleu, self).__init__()
        self.ngram = ngram
        self.bleu = Bleu(ngram=ngram).calc

    @property
    def name(self):
        return "self-bleu"

    def calc(self, hyps: list, *args, **kwargs):
        n = len(hyps)
        if n < 2:
            return 1
        scores = [self.bleu(hyps[i], hyps[j]) for i in range(n) for j in range(i + 1, n)]
        return sum(scores) / len(scores)

