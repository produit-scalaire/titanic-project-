import pandas as pd
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

#load data
df = pd.read_csv(("titanic/train.csv"))
#we remove the Name and Ticket colonnes and we transform str in float
df = df.drop("Name", axis = 1)
df = df.drop("Ticket", axis = 1)
df = df.drop("Cabin", axis = 1)
sex = {"male": 0, "female": 1}
df["Sex"] = df["Sex"].replace(sex)
embarked = {"C": 0, "Q": 1, "S": 2}
df["Embarked"] = df["Embarked"].replace(embarked)
print(df.values[0])

#neural network
class neural_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(neural_network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        return x

#we define the parameters of the net

input_size = 7
hidden_size = 64
output_size = 2


net = neural_network(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)

def Train(num_epochs, criterion, optimizer):
    n_total_steps = len(df.values)
    X_list = []
    Y_list = []
    for epoch in range (num_epochs):
        for i,passenger in enumerate(df.values):
            passenger = list(passenger)
            Y_train = [0, 0]
            if passenger.pop(1) == 1:
                Y_train[0] = 1
            else:
                Y_train[1] = 1
            passenger.pop(0)
            assert len(passenger) == 7
            Y_train = torch.Tensor(Y_train).to(dtype=torch.float32)
            X_train = torch.Tensor(passenger).to(dtype = torch.float32)
            Y_predict = net(X_train)
            loss = criterion(Y_predict, Y_train)
            X_list.append((i + 1) * (epoch + 1))
            Y_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
    print('Finished Training')
    return X_list, Y_list

X_list, Y_list = Train(1, criterion, optimizer)
PATH = './titanic_net.pth'
torch.save(net.state_dict(), PATH)

plt.figure("descente de gradient")
plt.plot(X_list, Y_list, color = "b")
plt.grid()
plt.show()

for