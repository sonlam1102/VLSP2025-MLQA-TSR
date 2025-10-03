import torch
from visual_bge.modeling import Visualized_BGE
import re
from tqdm import tqdm
from data import *
import pandas as pd
import numpy as np
import heapq
import json 


def embedding_articles(law_db, img_path, model):        
    model.eval()
    lst_id = []
    embedding = None
    
    for law in law_db:
        law_id = law['id']
        print(law_id)
        print(law['title'])
        print("---------=====---------")
        for d in tqdm(law['articles']):
            table_pattern = re.finditer(r"<<TABLE: \s*(.*?)\s* \/TABLE>>", d['text'], flags=re.DOTALL)
            if re.search(r"<<TABLE: \s*(.*?)\s* \/TABLE>>", d['text'], flags=re.DOTALL) is not None:
                print("process_table")
                for tbl in table_pattern:
                    table_str = tbl.group(1).strip()
                    md_table = pd.read_html(table_str, header=0)
                    md_table_to_md = md_table[0].to_markdown()
                    article = d['text'].replace(table_str, md_table_to_md)
                    d['text'] = article

        for d in tqdm(law['articles']):
            images_pattern = re.finditer(r"<<IMAGE: \s*(.*?)\s* \/IMAGE>>", d['text'])
            if re.search(r"<<IMAGE: \s*(.*?)\s* /IMAGE>>", d['text']) is not None:
                print("text+image")
                lst_imgs = []
                for img in images_pattern:
                    image_file = img.group(1).strip()
                    lst_imgs.append(img_path + "/" + image_file)
                    # article = re.sub(r"<<IMAGE: {} \/IMAGE>>".format(image_file), "", d['text'])
                    article = d['text'].replace("<<IMAGE: {} \/IMAGE>>".format(image_file), "")
                    d['text'] = article
                
                # print(lst_imgs)
                
                for i in range(0, len(lst_imgs)):
                    # lst_id.append("{}-{}".format(d['id'], str(i+1)))
                    lst_id.append("{}-{}_{}".format(d['id'], str(i+1), law_id))
                    # print(lst_imgs[i])
                    candidates = model.encode(image=lst_imgs[i],text=d['text'])
                    # lst_embedding.append(candidates)
                    if embedding is None:
                        embedding = candidates.cpu().detach().numpy()
                    else:
                        embedding = np.append(embedding, candidates.cpu().detach().numpy(), axis=0)
            else:
                print("text")
                candidates = model.encode(text=d['text'])
                lst_id.append(d['id']+"_"+law_id)
                if embedding is None:
                    embedding = candidates.cpu().detach().numpy()
                else:
                    embedding = np.append(embedding, candidates.cpu().detach().numpy(), axis=0)
        
    
    embedding_id = np.array(lst_id)
    print(embedding.shape)
    print(embedding_id.shape)
    
    return embedding_id, embedding


def get_top_k(predict_output, top_k):
    lst_out_one_hot = np.zeros(len(predict_output))
    idx_top_k = heapq.nlargest(top_k, range(len(predict_output)), predict_output.take)
    
    i = 1
    for idx in idx_top_k:
        lst_out_one_hot[idx] = i
        i = i + 1

    return lst_out_one_hot


def retrieve_articles(model, law_embeding, law_ids, image_path):
    def make_retrival(candidate):
        article_law = candidate.split("_")[0].split("-")[0]
        id_law = candidate.split("_")[1]
        
        return {
            "law_id": id_law,
            "article_id": article_law
        }
    
    with open("./dataset/vlsp_2025_private_test_retrieval_no_labels.json", "r", encoding="utf-8") as f:
        test = json.load(f)
    f.close() 
    
    lst_results = []
    for d in tqdm(test):
        relevant_ids = []
        
        img_path = image_path + "/" + d['image_id'] + ".jpg"
        query_emb = model.encode(image=img_path, text=d['question'])
        
        similarities = query_emb.cpu().detach() @ law_embeding.T
        
        lst_similars = similarities.T.detach().numpy()
        lst_oh = get_top_k(lst_similars, 5)
        assert len(lst_oh) == len(law_ids)
        for i in range (0, len(lst_oh)):
            if lst_oh[i] > 0:
                if make_retrival(law_ids[i]) not in relevant_ids:
                    relevant_ids.append(make_retrival(law_ids[i]))
        # d['predicted_relevant_articles'] = relevant_ids
        lst_results.append({
            "id": d['id'],
            "image_id": d['image_id'],
            "question": d["question"],
            "relevant_articles": relevant_ids
        })
        
    return lst_results


def eval_retrieval(y_pred, y_true):
    y_pred = {e['id']:set((p['law_id'],p['article_id']) for p in e['relevant_articles']) for e in y_pred}
    y_true = {e['id']:set((p['law_id'],p['article_id']) for p in e['relevant_articles']) for e in y_true}
    ids = list(y_true)

    precision = np.array([ len(y_true[k] & y_pred[k])/len(y_pred[k]) if y_pred.get(k) else 0 for k in ids])
    recall = np.array([ len(y_true[k] & y_pred.get(k,set()))/len(y_true[k]) for k in ids])
    f2 = (5 * precision * recall / (4 * precision + recall + 1e-9)).mean()

    return {'f2': f2}

def make_embedding():
    model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight="./Visualized_m3.pth")
    model.eval()
    
    law_db, image_path = read_db()
    
    law_id, law_embedding = embedding_articles(law_db, image_path, model)
    print(law_id.shape)
    print(law_embedding.shape)
    
    np.save("./law_id.npy", law_id)
    np.save("./law_embedding.npy", law_embedding)

def retrieval():
    model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight="./Visualized_m3.pth")
    model.eval()
    law_id = np.load("./law_id.npy")
    law_embedding = np.load("./law_embedding.npy")
    
    print(law_id.shape)
    print(law_embedding.shape)

    results = retrieve_articles(model, law_embedding, law_id, "./dataset/public_test_images")
    with open('public_test_task_1.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    
def eval():
    with open('public_test_task_1.json', 'r', encoding='utf-8') as f:
        prediction = json.load(f)
    f.close()
    
    with open('./dataset/vlsp_2025_public_test.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    f.close()
    
    print(eval_retrieval(prediction, ground_truth))

if __name__ == '__main__':
    make_embedding()
    retrieval()
    eval()