import json
from src import loadDataset

classes = ["ageism", "aporophobia", "body_shame", "capacitism", "lgbtphobia", "political", "racism", "religious_intolerance", "misogyny", "xenophobia"]

def select_exclusive_instances(dataset, target_class, n=3):
    
    filtered_ds = dataset.filter(lambda example: (sum(example.get(key, 0) for key in classes) == 1) and example[target_class] == 1)

    samples = filtered_ds.shuffle(seed=42).select(range(n))

    return samples.to_list()

def select_multi_instances(dataset, target_class, n=2):

    filtered_ds = dataset.filter(lambda example: (sum(example.get(key, 0) for key in classes) > 1) and example[target_class] == 1)

    samples = filtered_ds.shuffle(seed=42).select(range(n))

    return samples.to_list()

def main(): 
    train, test = loadDataset.loadTuPyE(-1, -1)
    dataset = test.add_column("id", list(range(len(test))))

    instances_dict = {}

    for c in classes:
        instances_dict[c] = {
            "exclusive": select_exclusive_instances(dataset, c, n=3),
            "multi": select_multi_instances(dataset, c, n=2)
        }

    with open("utils/selected_instances.json", "w", encoding='utf-8') as f:
        json.dump(instances_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()