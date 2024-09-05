
# %%
import jieba
from nltk.translate.bleu_score import sentence_bleu

  # source
target = 'the cat sat on the mat'  # target
inference = 'the cat is on the mat'  # inference

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import evaluate

predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party."]

references = ["It is a guide to action that ensures that the military will forever heed Party commands."]

bleu = evaluate.load("bleu")

results = bleu.compute(predictions=predictions, references=references)

print(results)

# # %%
# import evaluate
# import evaluate

# predictions = [inference]
# references = [target]
# predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party."]

# references = ["It is a guide to action that ensures that the military will forever heed Party commands."]

# bleu = evaluate.load("bleu")
# rouge = evaluate.load('rouge')

# results = rouge.compute(predictions=predictions, references=references)

# print(results)


# # 加载Distinct模型
# distinct = evaluate.load("lsy641/distinct")

# # 示例1：计算分数，提供了预测句子和vocab_size
# results1 = distinct.compute(predictions=["Hi.", "I am sorry to hear that", "I don't know", "Do you know who that person is?"], vocab_size=50257)

# # 打印结果
# print(results1)

# # 示例2：计算分数，提供了dataForVocabCal用于vocab_size的计算
# dataset = ["This is my friend jack", "I'm sorry to hear that", "But you know I am the one who always support you", "Welcome to our family","Hi.", "I am sorry to hear that", "I don't know", "Do you know who that person is?"]
# results2 = distinct.compute(predictions=["But you know I am the one who always support you", "Hi.", "I am sorry to hear that", "I don't know", "I'm sorry to hear that"], dataForVocabCal=dataset)
# # vocab_size的数据。通常应该是包含任务数据集的句子列表。要么提供vocab_size，要么提供dataForVocabCal中的数据。默认值为None。

# # 打印结果
# print(results2)


