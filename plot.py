import json
import matplotlib.pyplot as plt

# Charger le fichier trainer_state.json
log_file = "./qwen2.5-lora-output/trainer_state.json"
with open(log_file, "r") as f:
    state = json.load(f)

# Extraire les logs d'entraînement
logs = state["log_history"]
loss_values = [log["loss"] for log in logs if "loss" in log]
steps = list(range(1, len(loss_values) + 1))

# Tracer la courbe
plt.figure(figsize=(10, 5))
plt.plot(steps, loss_values, marker='o')
plt.title("Évolution de la loss pendant l'entraînement")
plt.xlabel("Étapes")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
