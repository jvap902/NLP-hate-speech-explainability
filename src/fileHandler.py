import json

def updateJson(json_path, fields, values, increment=None):
    with open(json_path, "r+") as f:
        json_data = json.load(f)
        
        for idx, field in enumerate(fields):
            if increment == None or field not in json_data:
                val = values[idx]
                
            else:
                val = json_data[field]+values[idx] if increment[idx] else values[idx]
                
            json_data[field] = val
        
        f.seek(0)
        json.dump(json_data, f, indent=4)
        f.truncate()

def getJsonInfo(json_path, fields=[]):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    
    if len(fields) == 0:
        return json_data
    
    data = []

    for field in fields:
        data.append(json_data[field])
        
    return data

def writeJson(json_path, dic):
    with open(json_path, "w") as f:
        json.dump(dic, f, indent=4)