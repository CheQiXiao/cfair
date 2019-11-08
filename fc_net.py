#!/user/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
#一个HDF5文件就是一个容器，用于储存两类对象：datasets，类似于数组的数据集合；groups，类似于文件夹的容器，可以储存datasets和其它groups。
from lr_utils import load_dataset
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
# index = 30
# print(train_set_x_orig[index])
# plt.imshow(train_set_x_orig[index])

#打印出当前的训练标签值
#使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
#print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
#只有压缩后的值才能进行解码操作
# print("train_set_y=" +str(train_set_y[:,index]))
# print(classes[np.squeeze(train_set_y[:,index])])
# plt.show()
#image.shape[0],image.shape[1],image.shape[2]表示图像长，宽，通道数  image.shape表示图片的维度
m_train=train_set_y.shape[1]
m_test=test_set_y.shape[1]
num_px=train_set_x_orig.shape[1]

print("训练集的数量：m_train="+str(m_train))
print("测试集的数量：m_test="+str(m_test))
print("每张图片的高和宽：num_px="+str(num_px))
print("每张图片的大小：（"+str(num_px)+","+str(num_px)+",3)")
print("训练集图片的维度："+str(train_set_x_orig.shape))
print("训练集标签的维度："+str(train_set_y.shape))
print("测试集图片的维度："+str(test_set_x_orig.shape))
print("测试集标签的维度："+str(test_set_y.shape))
#X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
#将训练集的维度降低并转置。这里的-1被理解为unspecified value，意思是未指定为给定的。如果我只需要特定的行数，列数多少我无所谓，我只需要指定行数，那么列数直接用-1代替就行了，计算机帮我们算赢有多少列，反之亦然。
#如果是reshape(5,-1) 就是将数组变为5行的矩阵，列的话根据具体的来分

train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
#
# print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
# print ("训练集_标签的维数 : " + str(train_set_y.shape))
# print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
# print ("测试集_标签的维数 : " + str(test_set_y.shape))
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    """

    :param z: 任意大小的标量或者numpy数组
    :return:
    """
    s=1/(1+np.exp(-z))
    return s

def initialize_with_zero(dim):
    """
    此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0,w b都被初始化为0
    :param dim:想要的w的大小
    :return:w-维度为（dim,1)的初始化向量  b-初始化的标量
    """
    w=np.zeros(shape=(dim,1))
    b=0
    assert (w.shape==(dim,1))#assert 表示如果出错则终止程序，断言函数是对表达式布尔值的判断，要求表达式计算值必须为真。如果表达式为假，触发异常；如果表达式为真，不执行任何操作。
    assert (isinstance(b,float)or isinstance(b,int))#isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
    return (w,b)

def propagate(w,b,X,Y):
    """

    :param w:权重，大小不等的数组（num_px * num_px * 3，1）
    :param b:偏差，一个标量
    :param X:矩阵类型为（num_px * num_px * 3，训练数量）
    :param Y: 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)
    :return:  cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """
    m=X.shape[1]  #X=np.array([[1,2,4,5], [3,4,6,1]]),X.shape[0]=2,X.shape[1]=4
    #正向传播
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))

    #反向传播
    dw=(1/m)*(np.dot(X,(A-Y).T))
    db=(1/m)*(np.sum(A-Y))
    # 使用断言确保我的数据是正确的
    assert (dw.shape==w.shape)
    assert (db.dtype==float)
    cost=np.squeeze(cost)#只有一行或一列的维度（a singleton dimension）被去除掉了

    assert (cost.shape==())
    grads={
        "dw":dw,
        "db":db
    }
    return (grads,cost)
# #测试一下propagate
# print("====================测试propagate====================")
# #初始化一些参数
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    此函数通过运行梯度下降算法来优化w和b

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值

    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。

    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        # 打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i, cost))

    params = {
            "w": w,
            "b": b}
    grads = {
            "dw": dw,
            "db": db}
    return (params, grads, costs)



# #测试optimize
# print("====================测试optimize====================")
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))


def predict(w, b, X):
    """
        使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，
        参数：
            w  - 权重，大小不等的数组（num_px * num_px * 3，1）
            b  - 偏差，一个标量
            X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据
        返回：
            Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）
        """
    m = X.shape[1]  # 图片的数量????为什么是图片的数量
    """
    shape函数是numpy.core.fromnumeric中的函数，它的功能是读取矩阵的长度，
    比如shape[0]就是读取矩阵第一维度的长度。
    shape的输入参数可以是一个整数（表示维度），也可以是一个矩阵。
    """
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)  # reshape函数：改变数组的维数

    # 计预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a[0,i]转换为实际预测p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        # 使用断言
    assert (Y_prediction.shape == (1, m))
    return Y_prediction

#
# #测试predict
# print("------测试predict------")
# w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
# print("predictions = " + str(predict(w,b,X)))
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本
    返回：
        d  - 包含有关模型信息的字典。
    """
    w, b = initialize_with_zero(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从字典“参数”中检索参数w和b
    w, b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")  # mean()函数功能：求取均值,np.abs()返回决定值
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")  # abs() 函数返回数字的绝对值

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d



print("------测试model------")
# 这里加载的是真实的数据，请参见上面的代码部分
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# 绘制图
costs = np.squeeze(d['costs'])
"""
squeeze（）函数的用法:
在机器学习和深度学习中，通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式[[]]），
如果直接利用这个数组进行画图可能显示界面为空（见后面的示例）。我们可以利用squeeze（）函数将表示向量
的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。
"""
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is:" + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                           print_cost=False)
    print('\n' + "--------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)  # loc:图例所有figure位置;shadow:控制是否在图例后面画一个阴影
# 设置图例legend背景颜色
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
