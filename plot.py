import json
import matplotlib.pyplot as plt

# Charger l’historique des logs
with open(f"trainer_state.json", "r") as f:
    state = json.load(f)

# Extraire les steps et la loss
logs = state["log_history"]
steps = [entry["step"] for entry in logs if "loss" in entry]
losses = [entry["loss"] for entry in logs if "loss" in entry]

# Tracer la courbe
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Training loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss during training")
plt.grid(True)
plt.legend()
plt.savefig(f"loss_curve.png")
plt.close()

