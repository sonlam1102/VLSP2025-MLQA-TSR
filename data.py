import json

def read_training():
    with open("./dataset/vlsp_2025_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    f.close()
    
    return data

def read_public():
    with open("./dataset/vlsp_2025_public_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    f.close()
    
    return data

def read_db():
    with open("./dataset/law_db/vlsp2025_law.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    f.close()
    
    return data, "./dataset/law_db/images"


if __name__ == '__main__':
    train_data = read_public()
    answer = []
    for d in train_data:
        answer.append(d['answer'])
    
    answer = list(set(answer))
    print(answer)
    
    print(json.dumps(train_data, indent=4))
    print(len(train_data))