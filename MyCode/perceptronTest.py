# 神经元模拟实现AND或者OR逻辑程序

class Perceptron:
    def __init__(self, input_para_num, acti_func):
        # 初始化激活函数
        self.activator = acti_func

        # 把所有参数W初始化都设置为0
        self.weights = [0.0 for _ in range(input_para_num)]

    def __str__(self):
        # 打印出所有的参数，按照W1, W2, b的顺序
        return repr(self.weights)

    def predict(self, row_vec):
        # 输入并得到预测的输出值
        act_values = 0.0

        for i in range(len(self.weights)):
            act_values += self.weights[i] * row_vec[i]

        return self.activator(act_values)

    def train(self, dataset, iteration, rate):
        # 训练函数，输入x1, x2和期望的输出值。
        # 设置训练的次数和学习率。
        for i in range(iteration):
            for input_vec_label in dataset:
                prediction = self.predict(input_vec_label)
                self._update_weights(input_vec_label, prediction, rate)

    def _update_weights(self, input_vec_label, prediction, rate):
        # 根据预测的输出值调整参数。
        # 计算损失值
        delta = input_vec_label[-1] - prediction
        
        # 根据损失值调整参数，学习率为rate。
        for i in range(len(self.weights)):
            self.weights[i] += rate * delta * input_vec_label[i]

def func_activator(input_value):
    # 激活函数
    return 1.0 if input_value >= 0.0 else 0.0

def get_training_dataset():
    # 输入的序列和期望输出值。
    # [偏置值，x1, x2, y]

    # dataset for AND
    dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]

    # dataset for OR
    #dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 1], [-1, 0, 1, 1]]

    return dataset

def train_and_perceptron():
    # 构建一个两个输入值的神经元
    # 包括x1, x2和偏置b
    p = Perceptron(3, func_activator)

    dataset = get_training_dataset()

    # 进行1000次训练，训练的学习率是0.01
    p.train(dataset, 1000, 0.01)

    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)

    print ('input [1, 1] = %d' % and_perception.predict([-1, 1, 1]))
    print ('input [0, 0] = %d' % and_perception.predict([-1, 0, 0]))
    print ('input [1, 0] = %d' % and_perception.predict([-1, 1, 0]))
    print ('input [0, 1] = %d' % and_perception.predict([-1, 0, 1]))
        
