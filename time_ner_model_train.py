# -*- coding: utf-8 -*-
'''
训练包含：TIME的中文NER任务模型
'''
import kashgari
print(kashgari.__version__)
from kashgari.corpus import DataReader
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

train_x, train_y = DataReader().read_conll_format_file('data/data_all/time.train')
valid_x, valid_y = DataReader().read_conll_format_file('data/data_all/time.dev')
test_x, test_y = DataReader().read_conll_format_file('data/data_all/time.test')

bert_embedding = BERTEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
                               task=kashgari.LABELING)

model = BiLSTM_CRF_Model(bert_embedding)
model.fit(train_x, train_y, valid_x, valid_y, batch_size=64, epochs=5)

model.save('models/time_ner.h5')

model.evaluate(test_x, test_y)