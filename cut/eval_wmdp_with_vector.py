import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch as t
import torch
import argparse
from cut.utils import load_model, forward_with_steering_vec
import json
import datetime

def prepare_data(data, batch_size=8):
    """
    Return a generator of batches of the form (text_batch, answers_batch)
    """
    batch = []
    for row in data:

        question = f"""\
The following are multiple choice questions (with answers).

{row['question']}
A. {row['choices'][0]}
B. {row['choices'][1]}
C. {row['choices'][2]}
D. {row['choices'][3]}
Answer:
"""
        ans = row['answer']
        batch.append((question, ans))
        if len(batch) == batch_size:
            yield batch
            batch = []

def get_accuracy(model, tokenizer, batches, steering_vec, module):
    # get token idxs for A, B, C, D
    A_idx = tokenizer.encode("A")[-1]
    B_idx = tokenizer.encode("B")[-1]
    C_idx = tokenizer.encode("C")[-1]
    D_idx = tokenizer.encode("D")[-1]
    choice_idxs = t.tensor([A_idx, B_idx, C_idx, D_idx]).to(model.device)

    corrects = []
    with t.no_grad():
        i = 0
        for batch in batches:
            texts = [x[0] for x in batch]
            answers = t.tensor([x[1] for x in batch]).to(model.device)
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

            # Add the steering vector at the specified layer
            outputs = forward_with_steering_vec(model, inputs, module=module, steering_vec=steering_vec, no_grad=True)
            outputs = outputs.logits
            # print(f'`get_accuracy:\n    {outputs=}')
            outputs = outputs[:, -1, choice_idxs]
            predictions = outputs.argmax(dim=-1)
            corrects.extend((predictions == answers).tolist())
            if i % 250 == 0:
                print(f'accuracy at batch {i}: {sum(corrects)/len(corrects)}')
            i += 1


    return corrects

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on a multiple choice dataset')
    parser.add_argument('--model_name_or_path', type=str, default="./models/HuggingFaceH4/zephyr-7b-beta_alpha-5000_batches-80_layer-7_2024-04-21-21-09-13")
    parser.add_argument('--data_path', type=str, default='data/wmdp-cyber.jsonl')
    parser.add_argument('--batch_size', type=int, default=4, help='The batch size to use for evaluation')
    parser.add_argument('--steering_vec_path', type=str, default ="steering_vectors_list_2024-04-16-21-31-47", help='The path to the steering vector file')
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )
    # parser.add_argument('--device', type=str, default='cuda:0', help='The device to use for evaluation, e.g. cuda, cpu')

    args = parser.parse_args()
    accs_over_coeffs = []
    range_s = 0.72 
    range_e = 0.74
    factor = 1000
    incr = 0.005
    factors_dict = dict()
    corrects_dict = dict()
    
    model, tokenizer = load_model(args.model_name_or_path)
    for i in range(int(range_s * factor), int(range_e * factor), int(incr * factor)):
        for j in range(6):
            reader = (json.loads(line) for line in open(args.data_path, "r"))
            batches = prepare_data(reader, args.batch_size)
            steering_lst = t.load(args.steering_vec_path)
            # print(f'{steering_lst=}\n{len(steering_lst)=}\n{len(steering_lst[0])=}')
            if j == 5:
                steering_vec = t.stack(steering_lst[0], dim=0)
                steering_vec = steering_vec.mean(dim=0)
                label = 'mean'
            else:
                steering_vec = steering_lst[0][j]
                label = f'{j}'
            steering_vec_coeff = i / factor
            steering_vec *= -1 * steering_vec_coeff
            print(f'steering vec coeff: -{steering_vec_coeff}')
            # print(f'{steering_vec.shape=},\n{steering_vec=}')
            module = eval(args.module_str.format(model_name="model", layer_id=args.layer_id))
            corrects = get_accuracy(model, tokenizer, batches, steering_vec, module)
            corrects_dict[label] = corrects
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            print(f'{corrects_dict=}')
            # t.save(corrects, f'corrects_wmdp_unlearned_with_steering_*-{steering_vec_coeff}_{date}')
            # print(f"\n\nAccuracy for steering vec *-{steering_vec_coeff}: {sum(corrects)/len(corrects) if len(corrects) > 0 else 0}\n")
            # accs_over_coeffs.append((accuracy, steering_vec_coeff))
        factors_dict[steering_vec_coeff] = corrects_dict
        print(f'{factors_dict=}')
        corrects_dict = dict()

    print(f'{factors_dict=}')
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    torch.save(factors_dict, f'factors_dict_({range_s}, {range_e})_{date}')
    # print(f'\n\n{accs_over_coeffs=}\n')
    # torch.save(factors_dict, f'factors_dict_({range_s}, {range_e})_{date}')



