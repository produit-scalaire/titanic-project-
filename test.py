import torch
import pandas as pd
import numpy as np
import neural_net

#load data
df = pd.read_csv("titanic/Train.csv")

#we remove the Name and Ticket colonnes and we transform str in float
df = df.drop("Name", axis = 1)
df = df.drop("Ticket", axis = 1)
df = df.drop("Cabin", axis = 1)
df = df.drop("PassengerId", axis = 1)
sex = {"male": 0, "female": 1}
df["Sex"] = df["Sex"].replace(sex)
embarked = {"C": 0, "Q": 1, "S": 2}
df["Embarked"] = df["Embarked"].replace(embarked)
df = df.dropna()
print(df.values[0])

input_size = 7
hidden_size = 64
output_size = 2


net = neural_net.neural_network(input_size, hidden_size, output_size)

# Charger le modèle sous forme de OrderedDict
model_state_dict = torch.load("titanic_net.pth")

# Créer une instance du modèle en utilisant sa classe
model = net  # Remplacez YourModel() par la classe de votre modèle

# Charger les poids du modèle à partir du state_dict
model.load_state_dict(model_state_dict)

# Mettre le modèle en mode d'évaluation
model.eval()

with torch.no_grad():

    n_correct = 0
    n_samples = len(df.values)

    for passenger in df.values:
        passenger = list(passenger)
        survive = passenger.pop(0)
        x = torch.tensor(passenger).to(dtype=torch.float32)
        output = model(x)
        index = np.argmax(output)
        if index != survive:
            n_correct +=  1
    accuracy = n_correct/n_samples
    print(f"accuracy of the net = {accuracy}")
    print(accuracy)
assert accuracy > 0.5




