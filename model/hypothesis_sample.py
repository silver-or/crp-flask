import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt


class Solution:
    def __init__(self) -> None:  
        self.X = [1, 2, 3]
        self.Y = [1, 2, 3]

        self.W = tf.placeholder(tf.float32)
        self.hypothesis = self.X * self.W

        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))
        self.sess = tf.Session()

        self.W_history = []
        self.cost_history = []

    def execute(self):
        tf.set_random_seed(777)

        for i in range(-30, 50):
            curr_W = i * 0.1
            curr_cost = self.sess.run(self.cost, {self.W: curr_W})
            self.W_history.append(curr_W)
            self.cost_history.append(curr_cost)

    def eval(self):
        # 차트로 확인
        plt.plot(self.W_history, self.cost_history)
        plt.show()

    def hook(self):
        self.execute()
        self.eval()

if __name__ == '__main__':
    tf.disable_v2_behavior()
    Solution().hook()
