
import torch
import torch.nn as nn
import torch.optim as optim

# Datos XOR
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

# Definición de la red neuronal
class XORNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


model = XORNet()

# Función de pérdida y optimizador
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Entrenamiento
for epoch in range(2000):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs, y)

    loss.backward()

    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluación
print("\nResultados:")

with torch.no_grad():
    predictions = model(X)

    for i in range(len(X)):
        pred = 1 if predictions[i].item() >= 0.5 else 0

        print(
            f"{X[i].numpy().astype(int)} -> "
            f"Probabilidad: {predictions[i].item():.4f} "
            f"Clase: {pred}"
        )
