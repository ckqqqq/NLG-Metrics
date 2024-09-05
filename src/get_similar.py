from tqdm import tqdm


from auto_metrics.bertScore import (
    BertScore
)

metrics=BertScore()
res=metrics.calc("你好啊","Fuck you")
print(res)

