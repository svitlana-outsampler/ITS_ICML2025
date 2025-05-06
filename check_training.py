import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Create the directory if it does not exist
output_dir = 'plotdiag'
os.makedirs(output_dir, exist_ok=True)

# Load sentence transformer model
model_name = "paraphrase-MiniLM-L6-v2"
print("Loading model " + model_name)
model_st = SentenceTransformer(model_name)

# Load a model trained for contradiction detection (NLI)
nli_model_name = "roberta-large-mnli"
print("Loading NLI model " + nli_model_name)
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def compute_semantic_similarity(gold_output, generated_output):
    emb_generated = model_st.encode([generated_output], convert_to_tensor=True)
    emb_gold = model_st.encode([gold_output], convert_to_tensor=True)

    cosine_score = torch.nn.functional.cosine_similarity(emb_generated, emb_gold).cpu().item()
    euclidean_score = euclidean_distances(emb_generated.cpu(), emb_gold.cpu())[0][0]

    return cosine_score, euclidean_score

def detect_contradiction_nli(premise, hypothesis):
    inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    #labels = ['entailment', 'neutral', 'contradiction']
    labels = ['contradiction', 'neutral', 'entailment'] 
    max_idx = torch.argmax(probs).item()
    return labels[max_idx], probs[0][max_idx].item()


# test the detection of contradiction
premise = "The cat is in the tree."
hypothesis = "The moggy is swimming."
label, score = detect_contradiction_nli(premise, hypothesis)
print(f"Label: {label}, Score: {score}")

# stop here for the moment
#exit()

# Load JSON files
with open('evaluation_before_training.json', 'r') as f:
    evaluation_avant = json.load(f)

with open('evaluation_final.json', 'r') as f:
    evaluation_apres = json.load(f)

assert len(evaluation_avant) == len(evaluation_apres), "Files do not have the same number of entries."

def plot_and_save(i):
    input_data = evaluation_avant[i]['input']
    series = json.loads(input_data.split('Series: ')[1])
    gold_output = evaluation_avant[i]['gold_output']
    diagnostic_avant = evaluation_avant[i]['generated_output']
    # cute the string to the first 600 characters
    diagnostic_avant = diagnostic_avant[:600]
    diagnostic_apres = evaluation_apres[i]['generated_output']

    print(f"Calculating similarity scores for entry {i}...")
    cosine_score_avant, euclidean_score_avant = compute_semantic_similarity(gold_output, diagnostic_avant)
    cosine_score_apres, euclidean_score_apres = compute_semantic_similarity(gold_output, diagnostic_apres)

    cass_label_avant, cass_score_avant = detect_contradiction_nli(gold_output, diagnostic_avant)
    cass_label_apres, cass_score_apres = detect_contradiction_nli(gold_output, diagnostic_apres)

    diagnostic_avant += f"\n\nCosine Score: {cosine_score_avant:.4f}\nEuclidean Distance: {euclidean_score_avant:.4f}"
    diagnostic_avant += f"\n\n[CASS] Relation: {cass_label_avant} (score: {cass_score_avant:.4f})"

    diagnostic_apres += f"\n\nCosine Score: {cosine_score_apres:.4f}\nEuclidean Distance: {euclidean_score_apres:.4f}"
    diagnostic_apres += f"\n\n[CASS] Relation: {cass_label_apres} (score: {cass_score_apres:.4f})"

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    axs[0].plot(series)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Time Series')

    axs[1].axis('off')
    axs[1].text(0.1, 0.5, gold_output, fontsize=10, verticalalignment='center', wrap=True, color='black')

    color_avant = 'red' if cass_label_avant == 'contradiction' else 'orange' if cass_label_avant == 'neutral' else 'blue'
    color_apres = 'red' if cass_label_apres == 'contradiction' else 'orange' if cass_label_apres == 'neutral' else 'green'

    axs[2].axis('off')
    axs[2].text(0.1, 0.5, diagnostic_avant, fontsize=10, verticalalignment='center', wrap=True, color=color_avant)

    axs[3].axis('off')
    axs[3].text(0.1, 0.5, diagnostic_apres, fontsize=10, verticalalignment='center', wrap=True, color=color_apres)

    fig.suptitle(f'Case number {i}', fontsize=16)
    file_name = os.path.join(output_dir, f'case_{i}.png')
    print(f"Saving figure to {file_name}")
    fig.savefig(file_name)
    plt.close(fig)

# Main interaction loop
while True:
    user_input = input(f"Enter the data number to plot (between 0 and {len(evaluation_avant) - 1}), 'a' for all, or 'q' to quit: ")

    if user_input.lower() == 'q':
        break
    elif user_input.lower() == 'a':
        for i in range(len(evaluation_avant)):
            plot_and_save(i)
    else:
        try:
            i = int(user_input)
            if 0 <= i < len(evaluation_avant):
                plot_and_save(i)
            else:
                print("Invalid data number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number, 'a' for all, or 'q' to quit.")
