# -*- coding: utf-8 -*-
'''
训练包含：ORG、LOC、PER的中文NER任务模型
'''

from kashgari.corpus import ChineseDailyNerCorpus

train_x, train_y = ChineseDailyNerCorpus.load_data('train')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('validate')
test_x, test_y = ChineseDailyNerCorpus.load_data('test')

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(valid_x)}")
print(f"test data count: {len(test_x)}", test_x[0], test_y[0])

import kashgari
from kashgari.embeddings import BERTEmbedding

bert_embed = BERTEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
                           task=kashgari.LABELING,
                           sequence_length=100)

from kashgari.tasks.labeling import BiLSTM_CRF_Model

# 还可以选择 `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`

model = BiLSTM_CRF_Model(bert_embed)
model.fit(train_x,
          train_y,
          x_validate=valid_x,
          y_validate=valid_y,
          epochs=20,
          batch_size=512)

model.save('models/org_loc_per_ner.h5')

model.evaluate(test_x, test_y)
