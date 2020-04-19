import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# XOR 문제를 해결하기 위해 dataset 만들기.
x_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).float()
y_data = torch.tensor([0, 1, 1, 0]).float()

"""
INPUT	OUTPUT
A	B	A XOR B
0	0	0
0	1	1
1	0	1
1	1	0
"""

class Model(nn.Module):
 
    def __init__(self, input_size, H1, output_size):
        '''
        # 1-Layer
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        '''
        
        # 2-Layer
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size,H1)
        self.fc2 = nn.Linear(H1, output_size)
        '''    
        # 4-layer
        self.fc1 = nn.Linear(input_size,H1)
        self.fc2 = nn.Linear(H1,H1)
        self.fc3 = nn.Linear(H1,H1)
        self.fc4 = nn.Linear(H1, output_size)
        '''

    def forward(self, x):
        '''
        # 1-layer
        x = F.sigmoid(self.fc1(x))
        return x
        '''
        # 2-layer  
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
        '''
        # 4-layer
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(self.fc4(x))
        return x
        '''
    def predict(self, x):
        return self.forward(x) >= 0.5



model = Model(2, 2, 1)

# 손실 함수 정의(BCE Loss)
loss_func = nn.BCELoss()

# optimizer(Adam)를 정의 (learning rate=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 2000
losses = []

# batch train step
for i in range(epochs):

    # forward path를 진행하고
    hypothesis = model.forward(x_data)
    # 손실을 loss에 저장
    loss = loss_func(hypothesis, y_data)
    # Batch gradient descent(BGD) => 매 스텝에서 훈련 데이터 전체를 모두 보고 update를 진행함. 

    print("epochs:", i, "loss:", loss.item())
    #매 스텝마다의 loss를 저장.
    losses.append(loss.item())

    # 옵티마이저 초기화
    optimizer.zero_grad()
    # 모델에 관련하여 손실의 경사를 계산한다. 
    loss.backward()
    # model optimizing
    optimizer.step()


def cal_score(X, y):
    y_pred = model.predict(X)
    score = float(torch.sum(y_pred.squeeze(-1) == y.byte())) / y.shape[0]

    return score


print('test score :', cal_score(x_data, y_data))
plt.plot(range(epochs), losses)
plt.show()


def plot_decision_boundray(X):
    x_span = np.linspace(min(X[:, 0]), max(X[:, 0]))
    y_span = np.linspace(min(X[:, 1]), max(X[:, 1]))

    xx, yy = np.meshgrid(x_span, y_span)

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()

    pred_func = model.forward(grid)

    z = pred_func.view(xx.shape).detach().numpy()

    plt.contourf(xx, yy, z)
    plt.show()


plot_decision_boundray(x_data)
