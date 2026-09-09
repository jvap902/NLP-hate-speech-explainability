import json
from collections import defaultdict
from src import loadDataset

classes = ['ageism', 'aporophobia', 'body_shame', 'capacitism', 'lgbtphobia', 'political', 'racism', 'religious_intolerance', 'misogyny', 'xenophobia', 'other']

def get_active_classes(row):
    active = ""
    
    for c in classes:
        if row[c] == 1:
            active += c + "_"
            
    return active.strip("_")

def class_statistics(dataset):
    
    dict = defaultdict(int)
    
    for row in dataset:
        active = get_active_classes(row)
        
        dict[active] += 1
    
    return dict

def main():
    train, test = loadDataset.loadTuPyE(-1, -1)
    
    dict = {}
    
    dict['train'] = class_statistics(train)
    dict['test'] = class_statistics(test)
    
    with open("utils/dataset_info.json", "w") as file:
        json.dump(dict, file, indent=4, sort_keys=True)
    
    
if __name__ == "__main__":
    main()