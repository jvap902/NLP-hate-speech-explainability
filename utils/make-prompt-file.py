import json
from src import loadDataset

def main():
    train, test = loadDataset.loadTuPyE(-1, -1)
    # 1. Add the column
    dataset = test.add_column("id", list(range(len(test))))

    # 2. Create the desired column order (id first, then all remaining columns)
    new_column_order = ["id"] + [col for col in dataset.column_names if col != "id"]

    # 3. Reorder the dataset columns
    dataset = dataset.select_columns(new_column_order)

    print(dataset.to_json("utils/json-prompt.json"))

if __name__ == "__main__":
    main()