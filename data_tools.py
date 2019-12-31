# -*- coding: utf-8 -*-
'''
数据拆分、合并、识别标注工具
'''
from kashgari.corpus import DataReader
import re
from tqdm import tqdm


def cut_text(text, lenth):
    textArr = re.findall('.{' + str(lenth) + '}', text)
    textArr.append(text[(len(textArr) * lenth):])
    return textArr


def clean_data(source_file, target_file, ner_model):
    
    data_x, data_y = DataReader().read_conll_format_file(source_file)

    with tqdm(total=len(data_x)) as pbar:
        for idx, text_array in enumerate(data_x):
            if len(text_array) <= 100:
                ners = ner_model.predict([text_array])
                ner = ners[0]
            else:
                texts = cut_text(''.join(text_array), 100)
                ners = []
                for text in texts:
                    ner = ner_model.predict([[char for char in text]])
                    ners = ners + ner[0]
                ner = ners         
            # print('[-----------------------', idx, len(data_x))
            # print(data_y[idx])
            # print(ner)
        
            for jdx, t in enumerate(text_array):
                if ner[jdx].startswith('B') or ner[jdx].startswith('I') :
                    if data_y[idx][jdx] == 'O':
                        data_y[idx][jdx] = ner[jdx]
           
            # print(data_y[idx])
            # print('-----------------------]')  
            pbar.update(1)
            
    f = open(target_file, 'a', encoding="utf-8")    
    for idx, text_array in enumerate(data_x):
        if idx != 0:
            f.writelines(['\n'])   
        for jdx, t in enumerate(text_array):
            text = t + ' ' + data_y[idx][jdx] 
            if idx == 0 and jdx == 0:
                text = text
            else:
                text = '\n' + text
            f.writelines([text])   
    
    f.close()   
    
    data_x2, data_y2 = DataReader().read_conll_format_file(source_file)
    print(data_x == data_x2, len(data_y) == len(data_y2), '数据清洗完成')              
