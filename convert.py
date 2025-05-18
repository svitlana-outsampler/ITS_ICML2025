import json

def convert_dataset_to_jsonl(input_path, output_path, truncate=1000):
    """
    Convertit un dataset au format jsonl en un jsonl input/output pour le fine-tuning LoRA.
    
    :param input_path: chemin vers le fichier JSONL d'origine.
    :param output_path: chemin où sauvegarder le fichier JSONL transformé.
    :param truncate: nombre maximal de points de samples.
    """
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f][:truncate]

    with open(output_path, 'w') as out_file:
        print("found ", len(data), " examples")
        for item in data:
            index = item["index"]
            series = item["series"]
            series_str = ', '.join(f"{x:02d}" for x in series)
            input_text = f"{item['question']} Series: [{series_str}]"
            # write output json object in a string
            output_text = json.dumps((item["description"]))

            jsonl_obj = {
                "index": index,
                "input": input_text,
                "output": output_text
            }

            out_file.write(json.dumps(jsonl_obj, ensure_ascii=False) + '\n')

    print(f"Conversion terminée : {output_path}")

# Exemple d'utilisation
convert_dataset_to_jsonl('dataset/data.jsonl', 'test_jsonl.jsonl', truncate=200)