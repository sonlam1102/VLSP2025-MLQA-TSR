import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model():
    model = AutoModel.from_pretrained(
        "5CD-AI/Vintern-3B-beta",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-3B-beta", trust_remote_code=True, use_fast=False)
    model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model

def answer_qa(model, tokenizer, image_path):
    with open("./dataset/vlsp_2025_public_test_qa_no_labels.json", "r", encoding="utf-8") as f:
        test = json.load(f)
    f.close()
    generation_config = dict(max_new_tokens=5, do_sample=False)
    lst_results = []
    
    for d in tqdm(test):
        img_pth = image_path + d['image_id'] + ".jpg"
        pixel_values = load_image(img_pth, max_num=12).to(torch.bfloat16).cuda()
        
        if d['question_type'] == "Multiple choice":
            choice = "A. " + str(d['choices']["A"]) + "\n" + "B. " + str(d['choices']["B"]) + "\n" + "C. " + str(d['choices']["C"]) + "\n" + "D. " + str(d['choices']["D"])
            question = f"""<image>\n {d['question']} \n {choice} \n Trả lời bằng một trong bốn đáp án: A, B, C hoặc D. Không giải thích thêm.
            """
        else:
            question = f"""<image>\n {d['question']} \n Trả lời bằng một trong hai đáp án: Đúng hoặc Sai. Không giải thích thêm.
            """
        # print(question)
        response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        lst_results.append({
            "id": d['id'],
            "image_id": d['image_id'],
            "question": d["question"],
            "question_type": d['question_type'],
            "relevant_articles": d['relevant_articles'],
            "answer": response
        })
    return lst_results
        

def eval_inference(y_pred, y_true):
    y_pred = {e['id']:e['answer'] for e in y_pred}
    y_true = {e['id']:e['answer'] for e in y_true}
    acc = np.mean([float(y_true[k] == y_pred.get(k)) for k in y_true])
    return {'accuracy':acc}

def qa():
    tokenizer, model = load_model()
    results = answer_qa(model, tokenizer, "./dataset/public_test_images/")
    with open('public_test_task_2.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()

def eval():
    with open('public_test_task_2.json', 'r', encoding='utf-8') as f:
        prediction = json.load(f)
    f.close()
    
    with open('./dataset/vlsp_2025_public_test.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    f.close()

    print(eval_inference(prediction, ground_truth))

if __name__ == "__main__":
    qa()
    eval()