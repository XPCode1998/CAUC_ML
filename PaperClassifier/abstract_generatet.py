from keybert import KeyBERT
from datasets import load_dataset

dataset = load_dataset(path='csv', data_files='data/paper_abstract.csv', split='train')
kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')


def calculate_evaluation_metrics(predicted_keywords, true_keywords):
    tp = len(set(predicted_keywords) & set(true_keywords))
    fp = len(set(predicted_keywords) - set(true_keywords))
    fn = len(set(true_keywords) - set(predicted_keywords))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


precision = 0
recall = 0
for data in dataset:
    keywords = kw_model.extract_keywords(data['Abstract'], keyphrase_ngram_range=(1, 2), stop_words='english',
                                         use_mmr=True, diversity=0.2)
    predicted_keywords = [keyword.lower() for keyword, _ in keywords]
    true_keywords = [keyword.strip().lower() for keyword in data['Author Keywords'].split(';')]
    precision_t, recall_t = calculate_evaluation_metrics(predicted_keywords, true_keywords)
    if precision_t>0.5:
        print(data['Abstract'])
        print(data['Author Keywords'])
        print(predicted_keywords)
    precision += precision_t
    recall += recall_t

precision /= len(dataset)
recall /= len(dataset)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
