# -*- coding: utf-8 -*-
# filename: main.py
import os
import sys
from flask import Flask,jsonify
import flask
import re
import jieba
from keras.layers import Embedding, Dense, Bidirectional, Conv1D, MaxPool1D, GRU,CuDNNGRU,BatchNormalization,Activation,Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
import Attention

class Region(object):
    def __init__(self,id,file):
        self.model_id = id
        self.tokenizer = joblib.load(os.path.join(file,'tokenizer_final.model'))
        self.lb = joblib.load(os.path.join(file,'lb.model'))
        self.model = self.cnn_rnn_attention(file)
        self.restr = r'[0-9\s+\.\!\/_,$%^*();?:\-<>《》【】+\"\']+|[+——！，；。？：、~@#￥%……&*（）]+'

    def cnn_rnn_attention(self,file):
        model = Sequential([
            Embedding(20000 + 1, 128, input_shape=(1000,)),
            Conv1D(128, 3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            # MaxPool1D(10),
            Bidirectional(GRU(128, return_sequences=True,reset_after=True), merge_mode='sum'),
            Attention(128),
            Dropout(0.5),
            Dense(self.lb.classes_.size, activation='softmax')
        ])
        model.load_weights(os.path.join(file,'model.h5'))
        return model

    def prdected(self, text):
        resu = text.replace('|', '').replace('&nbsp;', '').replace('ldquo', '').replace('rdquo',
                                                                                        '').replace(
            'lsquo', '').replace('rsquo', '').replace('“', '').replace('”', '').replace('〔', '').replace('〕', '')
        resu = re.split(r'\s+', resu)
        dr = re.compile(r'<[^>]+>', re.S)
        dd = dr.sub('', ''.join(resu))
        line = re.sub(self.restr, '', dd)
        seg_list = jieba.lcut(line)
        sequences = self.tokenizer.texts_to_sequences([seg_list])
        data = pad_sequences(sequences, maxlen=1000)
        pred = self.model.predict(data)
        return self.lb.inverse_transform(pred)
rootdir = sys.path[0]  # 此处是填写模型文件的根目录
models = []
citycode = os.listdir(rootdir)
for i in range(0, len(citycode)):
    path = os.path.join(rootdir, citycode[i])
    if os.path.isfile(path) == False:
        try:
            models.append(Region(path[-6:],path))
        except:
            pass
        # models[-1].prdected("北京北京") # 去掉报错
server=Flask(__name__)#__name__代表当前的python文件。把当前的python文件当做一个服务启动
@server.route('/city',methods=['post'])#只有在函数前加上@server.route (),这个函数才是个接口，不是一般的函数
def reg():
    content=flask.request.values.get('content')
    in_code = flask.request.values.get("in_code")
    id = ""
    for i in range(len(models)):
        if models[i].model_id == in_code:
            id = i
            break


    result = jsonify({"result":models[id].prdected(content)[0]})
    return  result

if __name__ == '__main__':
    server.run(
    host='0.0.0.0',
    port= 500,
    debug=True,
    use_reloader=False
    )
