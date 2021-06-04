# import
import numpy as np
import h5py
from random import randint  # Random initialize 를 위한 함수

# load MNIST dataset
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
# training dataset
# input handdigit data들이 들어있음, 이걸 numpy array로 다시 만들어준다는 코드
x_train = np.float32(MNIST_data['x_train'][:])
# 그 handdigit image에 대응되는 label
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
# test dataset
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
MNIST_data.close()


# convolution layer


class CONV_LAYER:

    # 초기화 method
    # dim_ifmap : dimension of input featuremap
    # num_inch : number of input channel
    # dim_kernel : dimension of kernel(filter) - width, height가 같다고 가정하고 하나만 적는다
    # num_outch : number of output channel
    # padding
    # stride는 1이라고 가정
    def __init__(self, dim_ifmap, num_inch, dim_kernel, num_outch, padding):

        # class의 member 변수로 지정
        self.dim_ifmap = dim_ifmap
        self.num_inch = num_inch
        self.dim_kernel = dim_kernel
        self.num_outch = num_outch
        self.padding = padding

        # weight들 정의 - random으로 initialize
        # kernel의 dimension은 4d(parameter 4개)
        self.kernels = np.random.rand(num_inch, num_outch, dim_kernel, dim_kernel) / \
            np.sqrt(num_inch * num_outch * dim_kernel * dim_kernel)

    # ppt 'CONV Forward' 참고
    def forward(self, ifmap):

        # 나누기 stride 는 나누기 1이므로 생략
        self.dim_ofmap = (
            self.dim_ifmap - self.dim_kernel + 2*self.padding) + 1

        # input featuremap padding
        padded_ifmap = np.pad(ifmap, ((
            0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        # output featuremap은 일단 0으로 initialize
        ofmap = np.zeros((self.num_outch, self.dim_ofmap,
                         self.dim_ofmap), dtype=float)
        for x in range(self.dim_ofmap):
            for y in range(self.dim_ofmap):
                for k in range(self.num_outch):
                    for c in range(self.num_inch):
                        for i in range(self.dim_kernel):
                            for j in range(self.dim_kernel):
                                ofmap[k, x, y] += self.kernels[c, k,
                                                               i, j] * padded_ifmap[c, x+i, y+j]
        return ofmap

    # backpropagation
    # ppt 'CONV Backpropagation' 참고
    # 필요한 파라미터 : dO(다음 layer에서 전달되어 오는 gradient), 현재 convolution layer에 input featuremap
    # dK를 구하기 위해선 I와 d0의 convolution
    def backprop(self, I, dO):

        # I는 padding 필요
        padded_I = np.pad(I, ((0, 0), (self.padding, self.padding),
                          (self.padding, self.padding)), 'constant')

        # dK는 output이므로 일단 zero로 initialize, dK는 4d (num_inch, num_outch, kernel의 width, kernel의 height)
        dK = np.zeros((self.num_inch, self.num_outch,
                      self.dim_kernel, self.dim_kernel), dtype=float)
        for x in range(self.dim_ofmap):
            for y in range(self.dim_ofmap):
                for k in range(self.num_outch):
                    for c in range(self.num_inch):
                        for i in range(self.dim_kernel):
                            for j in range(self.dim_kernel):
                                dK[c, k, i, j] += padded_I[c,
                                                           x+i, y+j] * dO[k, x, y]

        # dI 구현은 내가 알아서..^^ .....................
        # convolution layer를 하나 더 쌓는게 실습과제, 이 때 다음 convolution layer에 넘겨주는 값으로 dI가 필요

        return dK


# fully connected layer


class FC_LAYER:

    # 초기화 - input neuron의 개수, output neuron의 개수 받아서 초기화
    def __init__(self, num_in, num_out):
        self.kernel = np.random.randn(
            num_in, num_out) / np.sqrt(num_in * num_out)
        self.bias = np.random.randn(1, num_out) / np.sqrt(num_out)

    def forward(self, x):
        z = np.dot(x, self.kernel) + self.bias
        return z

    # ppt 'FC Backpropagation' 참고
    # 뒤에서부터 들어오는 보라색 gradient를 dZ2라고 하고
    # 현재 FC Layer의 input을 x라고 한다
    def backprop(self, x, dZ2):

        # dot product 순서 중요
        dW = np.dot(x.T, dZ2)
        dZ1 = np.dot(dZ2, self.kernel.T)
        dB = np.sum(dZ2, axis=0, keepdims=True)
        return dW, dZ1, dB


class RELU_LAYER:
    def forward(self, x):
        return x*(x > 0)

    def backprop(self, x):
        return 1.0*(x > 0)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# neural network로 예측한 값과 실제 ground truth의 차이를 loss function으로 만들어줘야 한다
# 그 loss function이 이 error function
class CROSS_ENTROPY_ERROR:
    def forward(self, x, y):
        return -1.0 * np.sum(np.multiply(np.log(x + 0.001e-10), y))
        # x + 0.001e-10 는 log0 일 때를 대비

    def backprop(self, x, y):
        return (x - y)

# 돌리기


# convolution layer
# handdigit image를 input으로 받는다
# MNIST dataset은 28*28 크기의 input image를 가지고 있음
# color image가 아니므로 channel size는 1
# kernel size는 3*3
conv1 = CONV_LAYER(dim_ifmap=28, num_inch=1,
                   dim_kernel=3, num_outch=5, padding=1)

# relu layer
relu1 = RELU_LAYER()

# fully connected layer
# input neuron의 개수 : convolution layer의 output = 28*28*5
# output neuron의 개수 : class의 개수, MNIST dataset의 handdigit recognition은 0-9 숫자 인식하므로 10
fc1 = FC_LAYER(28*28*5, 10)

# cross entropy layer
cse1 = CROSS_ENTROPY_ERROR()

lr = 0.001
num_epochs = 3
train_iterations = 1000  # 한번 epoch에 1000개의 이미지로 training 1000번 한다는 뜻
test_iteration = 100

for epoch in range(num_epochs):

    total_trained = 0  # 총 training 얼마나 했는지 출력해주는 변수
    train_correct = 0  # training할 때 답을 얼마나 맞췄는지 track해주는 변수
    train_cost = 0  # const function의 output을 track해주는 변수

    # training할 때 random하게 input image를 받기 위해서 씀
    rand_indices = np.random.choice(
        len(x_train), train_iterations, replace=True)

    for i in rand_indices:
        total_trained += 1
        # 지금은 minibatch size 1이지만 과제에선 2로 구현
        # x_train에서 i번째 image를 가져온다, 이 image는 하나의 벡터 784*1로 flatten 되어있으므로 1(channel size)*28*28인 3d input featuremap으로 reshape
        mini_x_train = x_train[i].reshape(1, 28, 28)
        mini_y_train = y_train[i]  # y는 label이 붙어있음
        one_hot_y = np.zeros((1, 10), dtype=float)
        one_hot_y[np.arange(1), mini_y_train] = 1.0

        # forward propagation
        conv1_ofmap = conv1.forward(mini_x_train)
        relu1_ofmap = relu1.forward(conv1_ofmap)
        # relu의 output은 5*28*28, 이를 fully connected layer의 input으로 사용하기 위해 1d로 flatten -> reshape
        fc1_out = fc1.forward(relu1_ofmap.reshape(1, 28*28*5))
        prob = softmax(fc1_out)  # 최종 output probability
        train_cost += cse1.forward(prob, one_hot_y)

        # backpropagation
        # cross entry에서 나온 gradient
        dCSE1 = cse1.backprop(prob, one_hot_y)
        # fully connected layer에서 나온 gradient 는 원래 input들 + 뒤에서 온 gradient dCSE1
        dW_FC1, dZ_FC1, dB_FC1 = fc1.backprop(
            relu1_ofmap.reshape(1, 28*28*5), dCSE1)
        dRELU1 = relu1.backprop(conv1_ofmap)
        dK_CONV1 = conv1.backprop(mini_x_train, np.multiply(
            dRELU1, dZ_FC1.reshape(5, 28, 28)))

        # weight update
        conv1.kernels -= lr * dK_CONV1
        fc1.kernel -= lr * dW_FC1
        fc1.bias -= lr * dB_FC1

        # forward했을 때 얼마만큼 답이 맞았는지 계산
        # probability가 가장 높은 게 답 : argmax를 써서 probability가 가장 높은 vector element의 index를 가져온다
        # 그 값이 mini_y_train과 같으면 정답
        train_correct += np.sum(np.equal(np.argmax(prob,
                                axis=1), mini_y_train))

        # 100개의 image를 training할 때 마다 출력
        if (total_trained % 100 == 0):
            print("Trained: ", total_trained, "/", train_iterations,
                  "\ttrain accuracy: ", train_correct/100, "\ttrain cost: ", train_cost/100)
            train_cost = 0
            train_correct = 0

    test_correct = 0
    for i in range(test_iteration):
        mini_x_test = x_test[i].reshape(1, 28, 28)
        mini_y_test = y_test[i]

        conv1_ofmap = conv1.forward(mini_x_test)
        relu1_ofmap = relu1.forward(conv1_ofmap)
        fc1_out = fc1.forward(relu1_ofmap.reshape(1, 28*28*5))
        prob = softmax(fc1_out)
        test_correct += np.sum(np.equal(np.argmax(prob, axis=1), mini_y_test))
    print("epoch #: ", epoch, "\ttest accuracy: ", test_correct/test_iteration)
