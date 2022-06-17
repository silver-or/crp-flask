import os
import sys
from unicodedata import name
from unittest import result
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
from sympy import im
from icecream import ic
import tensorflow.compat.v1 as tf

class CalculatorModel:
    def __init__(self) -> None:
        self.model = os.path.join(basedir, 'model')
        self.data = os.path.join(self.model, 'data')

    def calc(self, num1, opcode, num2):
        print(f'훅에 전달된 num1 : {num1}, opcode : {opcode}, num2 : {num2}')
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.import_meta_graph(self.model + '/calculator_' + opcode + '/model-1000.meta')  # 저장한 모델을 끌고 와서 tensorflow에 장착
            graph = tf.get_default_graph()
            w1 = graph.get_tensor_by_name('w1: 0')
            w2 = graph.get_tensor_by_name('w2: 0')
            feed_dict = {w1: float(num1), w2: float(num2)}
            op_to_restore = graph.get_tensor_by_name('op_' + opcode + ':0')
            result = sess.run(op_to_restore, feed_dict)
            print(f'최종 결과 : {result}')
        return result

    def create_add_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}  # 초기값 지정
        r = tf.add(w1, w2, name='op_add')
        sess = tf.Session()  # Session : 공간, ver2의 Sequential
        _ = tf.Variable(initial_value = 'fake_variable')  # Variable : 확률 변수
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # 머신을 저장함
        print(f"feed_dict['w1']) : {feed_dict['w1']}")
        print(f"feed_dict['w2']) : {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 덧셈 결과 : {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_add', 'model'), global_step=1000)

    def create_sub_model(self):
        pass

    def create_mul_model(self):
        pass

    def create_div_model(self):
        pass

if __name__ == '__main__':
    ic(basedir)
    tf.disable_v2_behavior()
    ic(tf.__version__)
    hello = tf.constant("Hello")
    session = tf.Session()
    ic(session.run(hello))
    c = CalculatorModel()
    c.create_add_model()