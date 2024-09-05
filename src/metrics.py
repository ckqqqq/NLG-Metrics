from tqdm import tqdm

from auto_metrics.rouges import (
    Rouge1,
    Rouge2,
    RougeL,
)
from auto_metrics.bleus import (
    Bleu,
    SelfBleu,
)
from auto_metrics.meteors import (
    Meteor,
)
from auto_metrics.perplexity import (
    Perplexity,
)
from auto_metrics.bertScore import (
    BertScore
)

class Metrics:

    AVAILABLE_METRICS = {
        "rouge-1": Rouge1,
        "rouge-2": Rouge2,
        "rouge-l": RougeL,
        "bleu": Bleu,
        "self-bleu": SelfBleu,
        "meteor": Meteor,
        "ppl": Perplexity,
        "bert-score": BertScore
    }
    DEFAULT_METRICS = ["bleu", "self-bleu", "meteor","rouge-1","rouge-2","rouge-l","bert-score"]

    def __init__(self, metrics:list=None, path:str=None):
        print(metrics)
        self.metrics = metrics or Metrics.DEFAULT_METRICS
        self.metrics = list((set(self.metrics) & set(Metrics.AVAILABLE_METRICS.keys())))
        self.models = list()
        for metric in self.metrics:
            if metric == "ppl":
                self.models.append(Metrics.AVAILABLE_METRICS[metric](model_path=path))
            else:
                self.models.append(Metrics.AVAILABLE_METRICS[metric]())

    def calc(self, inputs, verbose=False):
        results = list()
        for data in inputs:
            ref, hyps = data["ref"], data["hyps"]
            
            # print(type(ref))
            if type(ref)==list or type(hyps)==list:
                raise ValueError("")
            elif ref=="" or hyps=="":
                print(["空数据"]*10)
                
                continue
            else:
                if ref.startswith("Therapist:"):
                    ref=ref[len("Therapist:"):]
                if hyps.startswith("Therapist:"):
                    hyps=hyps[len("Therapist:"):]
            result_dict={
                metric: model.calc(hyps=hyps, ref=ref) for metric, model in zip(self.metrics, self.models)
            }
            print("\n",hyps,"\nXX",ref,"\n####",result_dict)
            results.append(result_dict)
        return results
