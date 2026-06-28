import torch
import torch.nn as nn

# Data
x = torch.tensor([[5.0]])  # 5 rupees
y = torch.tensor([[5.0]])  # correct answer

# ── 1. Network ────────────────────────────────────────────────────────
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)  # 1 input 1 output

    def forward(self, x):
        return self.layer1(x)

# ── 2. Setup ──────────────────────────────────────────────────────────
model     = MyNetwork()
loss_fn   = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"Initial weight: {model.layer1.weight.item():.4f}")
print(f"Initial bias:   {model.layer1.bias.item():.4f}")

# ── 3. Training loop ──────────────────────────────────────────────────
for epoch in range(100):
    prediction = model(x)
    loss       = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} — loss: {loss.item():.4f}  prediction: {prediction.item():.4f}")

# ── 4. Save ───────────────────────────────────────────────────────────
torch.save(model.state_dict(), "models/my_model.pt")
print("\nModel saved!")

# ── 5. Load and use ───────────────────────────────────────────────────
model.load_state_dict(torch.load("models/my_model.pt"))
print(f"Final prediction: {model(x).item():.4f}")
print(f"Correct answer:   {y.item()}")