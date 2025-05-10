from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
from transformers import TrainingArguments, Trainer

import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
dataset = load_dataset(path='csv', data_files='data/paper_classifier.csv', split='train')


def prepare_train_features(data):
    return tokenizer.batch_encode_plus(
        data['sentence'],
        padding='max_length',
        truncation=True,
        max_length=500,
    )


# 加载数据集
dataset = dataset.map(prepare_train_features, batched=True, batch_size=500, num_proc=4)
dataset = dataset.train_test_split(test_size=0.2)
dataset_train = dataset['train'].shuffle()
dataset_test = dataset['test'].shuffle()

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

# 加载评价函数
metric = load_metric('accuracy')


# 定义评价函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.argmax(axis=1)
    return metric.compute(predictions=logits, references=labels)


# 初始化训练参数
args = TrainingArguments(
    output_dir='./output_dir',
    learning_rate=1e-4,
    weight_decay=1e-2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
)

# 初始化训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
)

trainer.train()
