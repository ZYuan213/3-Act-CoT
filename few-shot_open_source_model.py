from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import pandas as pd
import re
import nltk
from tqdm import tqdm

device = "cuda"
model_type = 'llama2'
if model_type == 'llama2':
    model = AutoModelForCausalLM.from_pretrained('llama-2-7b-chat-hf',
                                                 torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('llama-2-7b-chat-hf')
elif model_type == 'gemma':
    model_path = 'gemma-7b-it'
    model = model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                         attn_implementation="flash_attention_2").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

print('=' * 100)
print(model_type)
print('=' * 100)

prompt = ''
examples = pd.read_csv('./example/en/ROC_example_8_en.csv')
for _, row in examples.iterrows():
    event = row['event'].replace('[EVENT_e]', '').split('[EVENT_sep]')
    event_sequence = f'1.{event[1]}2.{event[2]}3.{event[3]}4.{event[4]}'
    sentences = nltk.sent_tokenize(row['text'])
    cot_example = f"""Request:
1. Leading context: {row['leading_context']}
2. Event: {event_sequence}
Answer:
{' '.join(sentences[1:]).strip()}

"""
    prompt += cot_example
print(prompt)

test_dataset_path = './dataset/en/ROC_test_event_1.csv'
test_dataset = pd.read_csv(test_dataset_path)

COT_path_num = 20
data = {
    'index': [],
    'inputs': [],
    'label': []
}
for i in range(1, COT_path_num + 1):
    data[f'gen_story_{i}'] = []
    data[f'gen_text_{i}'] = []
save_df = pd.DataFrame(data)

temperature = 0.7
gen_result_path = f'./generate_result/{model_type}/few-shot/gen_200-600_completion_{temperature}.csv'
assert gen_result_path != ''
save_chunk_size = 1
begin = 0
print('*' * 100)
for index, row in test_dataset.iterrows():
    if index >= begin:
        print(f'第{index}行')
        event = row['event'].replace('[EVENT_e]', '').split('[EVENT_sep]')
        # event = f'1. {event[1].strip()} 2. {event[2].strip()} 3. {event[3].strip()} 4. {event[4].strip()}'
        event = f'1.{event[1]}2.{event[2]}3.{event[3]}4.{event[4]}'

        question = f"""Request:
1. Leading Context: {row['leading_context']}
2. Event: {event}
Answer:
"""
        print(question)
        input_text = prompt + question

        input_ids = tokenizer(input_text, return_tensors="pt").to(device)
        generated_texts = []
        gen_stories = []
        for _ in tqdm(range(COT_path_num)):
            output = model.generate(**input_ids, max_new_tokens=64, temperature=temperature, do_sample=True)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            generated_text = generated_text.replace(input_text, '')
            gen_story = generated_text.split('\n')[0].strip()

            generated_texts.append(generated_text)
            gen_stories.append(gen_story)
            print(gen_story)
            print('=' * 100)
        new_data = [{
            'index': str(index),
            'inputs': event + '\t' + row['leading_context'],
            'label': row['text'].replace(row['leading_context'], '').strip(),
        }]
        for i in range(COT_path_num):
            new_data[0][f'gen_story_{i + 1}'] = gen_stories[i]
            new_data[0][f'gen_text_{i + 1}'] = generated_texts[i]

        new_data = pd.DataFrame(new_data)
        save_df = pd.concat([save_df, new_data], ignore_index=True)
        if len(save_df) % save_chunk_size == 0 and (index + 1) == save_chunk_size:
            save_df.to_csv(gen_result_path, index=False, mode="a", encoding="utf_8_sig")
            save_df = save_df.drop(index=save_df.index)
            print("保存，行：" + str(index))

        elif len(save_df) % save_chunk_size == 0:
            save_df.to_csv(gen_result_path, index=False, mode="a", encoding="utf_8_sig", header=False)
            save_df = save_df.drop(index=save_df.index)
            print("保存，行：" + str(index))

if len(save_df) != 0:
    save_df.to_csv(gen_result_path, index=False, mode="a", encoding="utf_8_sig", header=False)
    save_df = save_df.drop(index=save_df.index)
    print("保存，行：" + str(index))


