import json

def convert_dataset_to_jsonl(input_path, output_path, truncate_series=1000):
    """
    Convertit un dataset au format json en un jsonl input/output pour le fine-tuning LoRA.
    
    :param input_path: chemin vers le fichier JSON d'origine.
    :param output_path: chemin où sauvegarder le fichier JSONL transformé.
    :param truncate_series: nombre maximal de points de la série à inclure (pour éviter des inputs trop longs).
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as out_file:
        print("found ", len(data), " examples")
        for item in data:
            index = item["index"]
            series = item["series"][:truncate_series]  # Tronque si nécessaire
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
convert_dataset_to_jsonl('dataset/data.json', 'test_jsonl.jsonl')