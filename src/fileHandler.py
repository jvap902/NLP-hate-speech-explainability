import os
import json
import csv

def createFile(file_path, content):
    
    if os.path.isfile(file_path):
        print("Arquivo já existente")
    else:
        print("Arquivo não existente, criando novo")
        with open(file_path, mode="a", newline='', encoding='utf-8') as f:
            f.write(content)

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
        if field in json_data.keys():
            data.append(json_data[field])
        else:
            data.append({}) #se campo não existir, vira dicionário vazio
        
    return data

def writeJson(json_path, dic):
    with open(json_path, "w") as f:
        json.dump(dic, f, indent=4)
        
def writeCsvLine(file_path, data, encoding='utf-8-sig'):
    with open(file_path, 'a', newline='', encoding=encoding) as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(data)

def findInCsv(file_path, params, values):
    if(len(params) != len(values)):
        raise ValueError("The number of parameters should be the as the number of values serched")
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        
        ans = []
        
        for row in reader:
            all_equal = True
            
            for idx, param in enumerate(params):
                if (str(values[idx]) != row[param]):
                    all_equal = False
                    break
            
            if all_equal:
                ans.append(row)
                
    return ans