from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from datasets import load_dataset
from sklearn.metrics import accuracy_score


dataset = load_dataset(path='csv', data_files='data/paper_classifier.csv', split='train')
dataset = dataset.train_test_split(test_size=0.2)
# 准备特征向量和标签
corpus = [data['sentence'] for data in dataset['train']]
labels = [data['label'] for data in dataset['train']]
# 特征提取
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
# 训练朴素贝叶斯分类器
classifier = MultinomialNB()
classifier.fit(X, labels)
test_data = dataset['test']['sentence']
test_label=dataset['test']['label']
# 对测试数据进行分类预测
X_test = vectorizer.transform(test_data)
predictions = classifier.predict(X_test)
accuracy = accuracy_score(test_label, predictions)
print("Accuracy:", accuracy)
# 打印预测结果
for i, prediction in enumerate(predictions):
    print(test_data[i], " --> ", prediction)



