
from matplotlib import pyplot as plt 

bread_price = [[0.5, 5], [0.6, 5.5], [0.8, 6], [1.1, 6.8], [1.4, 7]]

def BGD_step_gradient(w0_current, w1_current, points, learningRate):
    w0_gradient = 0
    w1_gradient = 0

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        w0_gradient += -1.0 * (y - ((w1_current * x) + w0_current))
        w1_gradient += -1.0 * x * (y - ((w1_current * x) + w0_current))

    new_w0 = w0_current - (learningRate * w0_gradient)
    new_w1 = w1_current - (learningRate * w1_gradient)

    return [new_w0, new_w1]

def gradient_descent_runner(points, start_w0, start_w1, l_rate, num_iterations):
    w0 = start_w0
    w1 = start_w1
    weight = [[w0, w1]]
    for i in range(num_iterations):
        w0, w1 = BGD_step_gradient(w0, w1, points, l_rate)
        # print('Iterate {}, w0={}, w1={}'.format(i, w0, w1))
        weight.append([w0, w1])

    return weight

def predict(w0, w1, wheat):
    price = w1 * wheat + w0
    return price

if __name__ == '__main__':
    learning_rate = 0.2
    num_iter = 100
    myweight = gradient_descent_runner(bread_price, 1, 1, learning_rate, num_iter)
    w0, w1 = myweight[-1]
    price = predict(w0, w1, 0.9)
    print("Price = ", price)

    w0_trend = []
    w1_trend = []

    for i in range(len(myweight)):
        w0_trend.append(myweight[i][0])
        w1_trend.append(myweight[i][1])

    plt.plot(range(len(w0_trend)), w0_trend, 'r')
    plt.plot(range(len(w1_trend)), w1_trend, 'g')
    plt.show()

