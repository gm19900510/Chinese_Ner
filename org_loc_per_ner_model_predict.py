# -*- coding: utf-8 -*-
'''
验证包含：ORG、LOC、PER的中文NER任务模型
'''
import kashgari
import re
kashgari.config.use_cudnn_cell = False
loaded_model = kashgari.utils.load_model('models/org_loc_per_ner.h5')


def cut_text(text, lenth):
    textArr = re.findall('.{' + str(lenth) + '}', text)
    textArr.append(text[(len(textArr) * lenth):])
    return textArr


def extract_labels(text, ners):
    ner_reg_list = []
    if ners:
        for word, tag in zip([char for char in text], ners):
            if tag != 'O':
                ner_reg_list.append((word, tag))

    # 输出模型的NER识别结果
    labels = {}
    if ner_reg_list:
        for i, item in enumerate(ner_reg_list):
            if item[1].startswith('B'):
                label = ""
                end = i + 1
                while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I'):
                    end += 1

                ner_type = item[1].split('-')[1]
   
                if ner_type not in labels.keys():
                    labels[ner_type] = []
                
                label += ''.join([item[0] for item in ner_reg_list[i:end]])
                labels[ner_type].append(label)    
                
    return labels


while True:
    text_input = input('sentence: ')
    ners = []
    texts = cut_text(text_input, 100)
    for text in texts:
        print(text)
        ner = loaded_model.predict([[char for char in text]])
        print(ner[0])
        ners = ners + ner[0]
    print(ners)
    labels = extract_labels(text_input, ners)
    print(labels)
    
