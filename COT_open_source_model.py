from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import pandas as pd
import re
import nltk
from tqdm import tqdm
import sys

device = "cuda"
model_type = 'llama2'
if model_type == 'llama2':
    model = AutoModelForCausalLM.from_pretrained('llama-2-7b-chat-hf',
                                                 torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('llama-2-7b-chat-hf')
elif model_type == 'gemma-7b':
    model_path = 'gemma-7b-it'
    model = model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                         attn_implementation="flash_attention_2").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

print('*' * 100)
print(model_type)
print('*' * 100)

prompt = ''
examples = pd.read_csv('./example/en/ROC_example_8_en.csv')
for index, row in examples.iterrows():
    event = row['event'].replace('[EVENT_e]', '').split('[EVENT_sep]')
    event_sequence = f'1.{event[1]}2.{event[2]}3.{event[3]}4.{event[4]}'
    sentences = nltk.sent_tokenize(row['text'])
    template = f"""Request:
1. Leading Context: {row['leading_context']}
2. Event: {event_sequence}
Answer:
A. Event:
    The following actions will appear in the story next: {event_sequence}
    The "setup" of the story includes event: {event[1].strip()}
    The "confrontation" of the story includes events: 1. {event[2].strip()} 2. {event[3].strip()}
    The "resolution" of the story includes event: {event[4].strip()}
B. Setup:
    The content of the story is related to leading context: "{row['leading_context']}".
    "{event[1].strip()}": {sentences[1]}
C. Confrontation:
    "{event[2].strip()}": {sentences[2]}
    "{event[3].strip()}": {sentences[3]}
D. Resolution:
    "{event[4].strip()}": {sentences[4]}
E. Story:
Combining the setup, confrontation, and resolution, the story is: {' '.join(sentences[1:])}

"""
    prompt += template
print(prompt)

test_dataset_path = './dataset/en/ROC_test_event_1.csv'
test_dataset = pd.read_csv(test_dataset_path)

data = {
    'index': [],
    'inputs': [],
    'gen_story': [],
    'label': [],
    'gen_text': []
}
save_df = pd.DataFrame(data)
"""
参数
"""
temperature = 1
gen_result_path = f'generate_result/{model_type}/COT/gen_200_completion_{temperature}.csv'

print('*' * 100)
for index, row in test_dataset.iterrows():
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

    # input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # output = model.generate(input_ids, num_return_sequences=1, max_new_tokens = 512, temperature = 0.5, eos_token_id=tokenizer.eos_token_id, do_sample = True)
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    for _ in range(10):
        output = model.generate(**input_ids, max_new_tokens=512, temperature=temperature, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        generated_text = generated_text.replace(input_text, '')
        gen_story = re.search(r'Combining the setup, confrontation, and resolution, the story is: [\s\S]*',
                              generated_text)
        if gen_story != None:
            break
    try:
        gen_story = \
        gen_story.group().replace('Combining the setup, confrontation, and resolution, the story is: ', '').split('\n')[
            0].strip().replace(row['leading_context'].strip(), '').strip()
    except AttributeError as e:
        print(e)
        print(generated_text)
        sys.exit()
    """
    释放显存
    """
    fake_input = tokenizer('X', return_tensors="pt").to("cuda")
    output = model.generate(**input_ids, max_new_tokens=1, temperature=0.5, do_sample=True)
    del fake_input
    del output
    torch.cuda.empty_cache()

    print(gen_story)
    print('=' * 100)

    new_data = [{
        'index': str(index),
        'inputs': event + '\t' + row['leading_context'],
        'gen_story': gen_story,
        'label': row['text'].replace(row['leading_context'], '').strip(),
        # 'gen_text':response['content']
        'gen_text': generated_text
    }]
    new_data = pd.DataFrame(new_data)
    save_df = pd.concat([save_df, new_data], ignore_index=True)

save_df.to_csv(gen_result_path, index=False, mode="w", encoding="utf_8_sig")


