# -*- coding: utf-8 -*-
'''
验证包含：TIME的中文NER任务模型
'''
import kashgari

loaded_model = kashgari.utils.load_model('models/time_ner.h5')


def extract_times(text, _y):
    ner_reg_list = []
    if t:
        for word, tag in zip([char for char in text], _y[0]):
            if tag != 'O':
                ner_reg_list.append((word, tag))

    # 输出模型的NER识别结果
    times = []
    if ner_reg_list:
        for i, item in enumerate(ner_reg_list):
            if item[1].startswith('B'):
                time = ""
                end = i + 1
                while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I'):
                    end += 1

                time += ''.join([item[0] for item in ner_reg_list[i:end]])
                times.append(time)    
                
    return times


while True:
    text = input('sentence: ')
    t = loaded_model.predict([[char for char in text]])
    print(t)
    times = extract_times(text, t)

    for t in times:
        print(t)    
