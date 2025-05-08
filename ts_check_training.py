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
    labels = ['no', 'bof', 'ok'] 
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

import ast 
import re
def plot_and_save(i):
    input_data = evaluation_avant[i]['input']
    series_str = input_data.split('Series: ')[1].strip()
    series_str = re.sub(r'\b0+(\d)', r'\1', series_str)
    series = ast.literal_eval(series_str)
    #series = json.loads(input_data.split('Series: ')[1])
    #exit()
    gold_output = evaluation_avant[i]['gold_output']
    diagnostic_avant = evaluation_avant[i]['generated_output']
    # cute the string to the first 600 characters
    diagnostic_avant = diagnostic_avant[:600]
    diagnostic_apres = evaluation_apres[i]['generated_output']

    sentence_gold = []
    sentence_after = []

    # extract json object from the string before
    json_gold = json.loads(gold_output)
    sentence_gold.append(json_gold['trend'])
    sentence_gold.append(json_gold['noise'])
    sentence_gold.append(json_gold['extrema'])

    json_after = json.loads(diagnostic_apres)
    sentence_after.append(json_after['trend'])
    sentence_after.append(json_after['noise'])
    sentence_after.append(json_after['extrema'])


    # Compute similarity scores for each sentence
    print(f"Calculating similarity scores for entry {i}...")
    cosine_score_gold, euclidean_score_gold = [] , []
    cosine_score_apres, euclidean_score_apres = [] , []
    cass_label_gold, cass_score_gold = [] , []
    cass_label_apres, cass_score_apres = [] , []
    # for each sentence in the json object
    for iss in range(len(sentence_gold)):
        a , b = compute_semantic_similarity(sentence_after[iss], sentence_gold[iss])
        print(sentence_after[iss], sentence_gold[iss], a,b)
        cosine_score_apres.append(a)
        euclidean_score_apres.append(b)
        a,b = detect_contradiction_nli(sentence_after[iss], sentence_gold[iss])
        cass_label_apres.append(a)
        cass_score_apres.append(b)

    # Concatenate the similarity scores and CASS labels properly
    for cosine, euclidean, cass_label, cass_score in zip(cosine_score_apres, euclidean_score_apres, cass_label_apres, cass_score_apres):
        diagnostic_apres += f"\n\nCosine Score: {cosine:.4f}\nEuclidean Distance: {euclidean:.4f}"
        diagnostic_apres += f"\n\n[CASS] Relation: {cass_label} (score: {cass_score:.4f})"


    fig, axs = plt.subplots(1+3, 1, figsize=(10, 12))
    axs[0].plot(series)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Time Series')

    for iss in range(len(sentence_gold)):
        axs[1+iss].axis('off')
        disp = 'gold: '+sentence_gold[iss] + ' \n' + 'after: ' + sentence_after[iss] + ' \n' + 'cosine: ' + str(cosine_score_apres[iss]) + ' \n' + 'euclidean: ' + str(euclidean_score_apres[iss]) + ' \n' + 'cass: ' + cass_label_apres[iss] + ' \n' + 'cass score: ' + str(cass_score_apres[iss])
        color = 'red' if cass_label_apres[iss] == 'no' else 'orange' if cass_label_apres[iss] == 'bof' else 'green'
        axs[1+iss].text(0.1, 0.5, disp, fontsize=10, verticalalignment='center', wrap=True, color=color)

    # color_avant = 'red' if cass_label_gold == 'contradiction' else 'orange' if cass_label_gold == 'neutral' else 'blue'
    # color_apres = 'red' if cass_label_apres == 'contradiction' else 'orange' if cass_label_apres == 'neutral' else 'green'

    # axs[2].axis('off')
    # axs[2].text(0.1, 0.5, diagnostic_avant, fontsize=10, verticalalignment='center', wrap=True, color=color_avant)

    # axs[2].axis('off')
    # axs[2].text(0.1, 0.5, diagnostic_apres, fontsize=10, verticalalignment='center', wrap=True, color=color_apres)

    fig.suptitle(f'Case number {i}', fontsize=16)
    file_name = os.path.join(output_dir, f'case_{i}.png')
    print(f"Saving figure to {file_name}")
    fig.savefig(file_name)
    plt.close(fig)
    scores = [0 if ca == 'no' else 1 if ca == 'ok' else 0.5 for ca in cass_label_apres]
    return scores


# Main interaction loop
while True:
    user_input = input(f"Enter the data number to plot (between 0 and {len(evaluation_avant) - 1}), 'a' for all, or 'q' to quit: ")

    if user_input.lower() == 'q':
        break
    elif user_input.lower() == 'a':
        mean_scores = [0.,0.,0.]
        for i in range(len(evaluation_avant)):
            scores = plot_and_save(i)
            print(f"Scores for case {i}: {scores}")
            for j in range(3):
                mean_scores[j] += scores[j]
        for j in range(3):
            mean_scores[j] /= len(evaluation_avant)
        print(f"Mean scores: {mean_scores}")
    else:
        try:
            i = int(user_input)
            if 0 <= i < len(evaluation_avant):
                scores = plot_and_save(i)
                print(f"Scores for case {i}: {scores}")
            else:
                print("Invalid data number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number, 'a' for all, or 'q' to quit.")
