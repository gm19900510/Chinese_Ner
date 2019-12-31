# -*- coding: utf-8 -*-
'''
训练包含：ORG、LOC、PER、TIME的中文NER任务模型
'''

import kashgari
from kashgari.corpus import DataReader
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari import utils

kashgari.config.use_cudnn_cell = False

train_x, train_y = DataReader().read_conll_format_file('data/data_all/example.train')
valid_x, valid_y = DataReader().read_conll_format_file('data/data_all/example.dev')
test_x, test_y = DataReader().read_conll_format_file('data/data_all/example.test')

train_x, train_y = utils.unison_shuffled_copies(train_x, train_y)
valid_x, valid_y = utils.unison_shuffled_copies(valid_x, valid_y)
test_x, test_y = utils.unison_shuffled_copies(test_x, test_y)

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(valid_x)}")
print(f"test data count: {len(test_x)}", test_x[0], test_y[0])

bert_embedding = BERTEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
                               task=kashgari.LABELING,
                               sequence_length=100)

model = BiLSTM_CRF_Model(bert_embedding)
model.fit(train_x, train_y, valid_x, valid_y, batch_size=512, epochs=20)

model.save('models/all_ner.h5')

model.evaluate(test_x, test_y)
