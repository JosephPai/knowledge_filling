import torch
import pdb
import json
from transformers import BertLMHeadModel, BertForMaskedLM, BertConfig, EncoderDecoderModel
from transformers import AdamW
from transformers import BertTokenizer


def dict2sent(d):
    ret = []
    for k in d.keys():
        ret.extend(d[k])
    ret = ", ".join(ret)
    return ret


# load data
print('loading data')
with open('semart_knowledge_filling_dataset/TRAIN_data.json', 'r') as f:
    train_data = json.load(f)
with open('semart_knowledge_filling_dataset/VAL_data.json', 'r') as f:
    val_data = json.load(f)
with open('semart_knowledge_filling_dataset/TEST_data.json', 'r') as f:
    test_data = json.load(f)

# set parameters
batch_size = 2
epochs = 2
lr = 1e-5

# init tokenizer and model
print('init model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

config = BertConfig.from_pretrained("bert-base-cased")
config.is_decoder = True
config.return_dict = True
model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config).cuda()
optimizer = AdamW(model.parameters(), lr=lr)
print(model)

# training
print('start train')
model.train()
batch_num = len(train_data) // batch_size
for epoch in range(epochs):
    data_idx = 0
    while data_idx + batch_size < len(train_data):
        data_batch = train_data[data_idx: data_idx + batch_size]
        data_idx += batch_size

        # organize data into pair and output seq
        batch_sent1, batch_sent2, batch_label = [], [], []
        for i in range(batch_size):
            batch_sent1.append(str(data_batch[i][0]).replace('<end>', '').strip())
            batch_sent2.append(dict2sent(data_batch[i][1]))
            batch_label.append(str(data_batch[i][2]).replace('<end>', '').strip())

        encoding = tokenizer(batch_sent1, batch_sent2, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()
        labels = tokenizer(batch_label, return_tensors='pt', padding='max_length', truncation=True,
                           max_length=int(input_ids.size(1)))['input_ids'].cuda()

        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        print('Epoch: [{}], Batch: [{}/{}], Loss:{}'.format(epoch, data_idx//batch_size, batch_num, loss))

model.save_pretrained('model_ckpt/')

