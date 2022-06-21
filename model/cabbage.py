from importlib.resources import path
import os
import sys
from turtle import shape
from unicodedata import name
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
from icecream import ic

import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd


class CabbageModel():
    def __init__(self) -> None:
        self.path = './data/price_data.csv'
        self.df = None
        self.basedir = os.path.join(basedir, 'model')
        self.x_data = None
        self.y_data = None

    def result(self, avg_temp, min_temp, max_temp, rain_fall):
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = [avg_temp, min_temp, max_temp, rain_fall]
            print(f'최종 결과 : {result}')
        return result

    def preprocessing(self):
        self.df = pd.read_csv(self.path, encoding='UTF-8', thousands=',')
        ic(self.df.head(5))
        # avgTemp, minTemp, maxTemp, rainFall, avgPrice
        xy = np.array(self.df, dtype=np.float32)  # csv 파일을 배열로 변환
        ic(type(xy))    # <class 'numpy.ndarray'>
        ic(xy.ndim)     # xy.ndim: 2  # 차원
        ic(xy.shape)    # xy.shape: (2922, 6)  # 행렬의 갯수
        self.x_data = xy[:, 1:-1]  # 피쳐
        self.y_data = xy[:, [-1]]  # 답

    def create_model(self):  # 모델 생성
        # 텐서 모델 초기화 (모델 템플릿 생성)
        # model = tf.global_variables_initializer()
        # 확률변수 데이터 (price_data)
        self.preprocessing()
        # 선형식 (가설) 제작 (y = Wx + b)
        X = tf.placeholder(tf.float32, shape=[None, 4])  # placeholder : 외부에서 주입되는 값
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name="weight")  # X 값에 의해 계속 변형, 하나가 결정됨
        b = tf.Variable(tf.random_normal([1]), name="bias")
        hypothesis = tf.matmul(X, W) + b  # 가설 : 선형식 (Wx + b)
        # 손실함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 예측값 - 실제값
        # 최적화 알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        # 세션 생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # _ = tf.Variable(initial_value = 'fake_variable')  # Variable : 확률 변수
        
        # 트레이닝
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                        feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0:
                print('# %d 손실 비용 : %d'%(step, cost_))
                print('- 배추 가격 : %d'%(hypo_[0]))

        # 모델 저장
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'), global_step=1000)
        print('저장 완료')

    def load_model(self, avgTemp, minTemp, maxTemp, rainFall):  # 모델 로드
        tf.disable_v2_behavior()
        # 선형식 (가설) 제작 (y = Wx + b)
        X = tf.placeholder(tf.float32, shape=[None, 4])  # placeholder : 외부에서 주입되는 값
        W = tf.Variable(tf.random_normal([4, 1]), name="weight")  # X 값에 의해 계속 변형, 하나가 결정됨
        b = tf.Variable(tf.random_normal([1]), name="bias")

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt-1000'))
            data = [[avgTemp, minTemp, maxTemp, rainFall]]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print(dict)
        return int(dict[0])

if __name__ == '__main__':
    tf.disable_v2_behavior()
    CabbageModel().create_model()
    