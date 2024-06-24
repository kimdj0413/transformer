import transformers
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import urllib.request
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/new_list_result.csv', encoding='cp949')
data.drop_duplicates(subset=['headline'], inplace=True)
data = data.dropna(how = 'any')
train_data, test_data = train_test_split(data, test_size=0.2)
train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

headline_lengths = [len(headline) for headline in data['headline']]
average_length = sum(headline_lengths) / len(headline_lengths)
max_length = max(headline_lengths)
# print(f"Headline 평균 길이: {average_length}")
# print(f"Headline 최대 길이: {max_length}")

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

max_seq_len = 128

# encoded_result = tokenizer.encode("하이닉스, D램 공급과잉 축소 기대 6일째↑", truncation=True, max_length=max_seq_len, padding='max_length')

def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):
    
    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []
    
    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        input_id = tokenizer.encode(example, truncation=True, padding='max_length', max_length=max_seq_len)
        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
        token_type_id = [0] * max_seq_len

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels

train_X, train_y = convert_examples_to_features(train_data['headline'], train_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)
test_X, test_y = convert_examples_to_features(test_data['headline'], test_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)

input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
max_seq_len = 128
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1]
        prediction = self.classifier(cls_token)

        return prediction
model = TFBertForSequenceClassification("klue/bert-base")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])

model.fit(train_X, train_y, epochs=2, batch_size=8, validation_split=0.2)
results = model.evaluate(test_X, test_y, batch_size=8)
print("test loss, test acc: ", results)

def finance_predict(new_sentence):
  input_id = tokenizer.encode(new_sentence, max_length=max_seq_len, pad_to_max_length=True)

  padding_count = input_id.count(tokenizer.pad_token_id)
  attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
  token_type_id = [0] * max_seq_len

  input_ids = np.array([input_id])
  attention_masks = np.array([attention_mask])
  token_type_ids = np.array([token_type_id])

  encoded_input = [input_ids, attention_masks, token_type_ids]
  score = model.predict(encoded_input)[0][0]

  if(score > 0.5):
    print("{:.2f}% 확률로 상승입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 하락입니다.\n".format((1 - score) * 100))

finance_predict("48단서 단숨에 72단으로…추격자 SK의 `적층신공`")  # 상승
finance_predict("숨고르기 속 ‘저점매수’ 전략...스탁론으로 싸게 담을 종목은?") # 하락
finance_predict('“삼성전자, 2Q 영업익 8.4조 예상…경쟁력 회복 기대”')  # 상승
finance_predict("‘중산층 세금’ 돼버린 ‘부자 세금’…野몽니에도 상속세 개편 추진")   # 하락