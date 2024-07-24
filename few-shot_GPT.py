import openai
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import nltk
openai.api_key = "Your_OpenAI_key"
# openai.api_base = ""

prompt = ''

examples = pd.read_csv('example/en/ROC_example_8_en.csv')
for _, row in examples.iterrows():
    event = row['event'].replace('[EVENT_e]', '').split('[EVENT_sep]')
    event = f'1. {event[1].strip()} 2. {event[2].strip()} 3. {event[3].strip()} 4. {event[4].strip()}'
    sentences = nltk.sent_tokenize(row['text'])
    cot_example = f"""Request:
1. Leading context: {row['leading_context']}
2. Event: {event}
Answer:
{' '.join(sentences[1:]).strip()}

"""
    prompt += cot_example

print(prompt)


def gpt_generation_completion(messages):
    completion = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=messages,
        temperature=0.2,
        max_tokens=1024
    )
    return completion.choices[0].text

test_dataset_path = 'dataset/en/ROC_test_event_1.csv'
gen_result_path = 'gen_result/gen_result/few-shot_gen_result.csv'

test_dataset = pd.read_csv(test_dataset_path)
test_dataset = Dataset.from_pandas(test_dataset)
index = 0

COT_path_num = 1
data = {
    'index': [],
    'inputs': [],
    'label': []
}
for i in range(1,COT_path_num+1):
    data[f'gen_story_{i}'] = []
    data[f'gen_text_{i}'] = []
save_df = pd.DataFrame(data)

save_chunk_size = 1
begin = 0
for example in test_dataset:
    if index >= begin:
        print(f'第{index}行')
        gen_stories = []
        gen_texts = []
        try:
            event = example['event'].replace('[EVENT_e]', '').split('[EVENT_sep]')
            event = f'1.{event[1]}2.{event[2]}3.{event[3]}4.{event[4]}'

            question = f"""Request:
1. Leading Context: {example['leading_context']}
2. Event: {event}
"""

            input_text = prompt + question
            print(question)

            for _ in tqdm(range(COT_path_num)):

                for _ in range(5):
                    try:
                        response = gpt_generation_completion(input_text)
                        gen_story = response
                        if gen_story != None:
                            break
                    except openai.error.AuthenticationError as e:
                        print("="*100)
                        print(e)
                    except openai.error.RateLimitError as e:
                        print("="*100)
                        print(e)
                    except openai.error.APIError as e:
                        print("="*100)
                        print(e)
                    except openai.error.ServiceUnavailableError as e:
                        print("="*100)
                        print(e)
                gen_story = gen_story.replace('\n', '').strip()
                gen_stories.append(gen_story)
                gen_texts.append(response)
        except AttributeError as e:
            print(e)
            print(response)

        new_data = [{
            'index': str(index),
            'inputs': event + '\t' + example['leading_context'],
            'label':example['text'].replace(example['leading_context'], '').strip()
        }]
        for i in range(COT_path_num):
            new_data[0][f'gen_story_{i+1}'] = gen_stories[i]
            new_data[0][f'gen_text_{i+1}'] = gen_texts[i]

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
    index += 1

if len(save_df) != 0:
    save_df.to_csv(gen_result_path, index=False, mode="a", encoding="utf_8_sig", header=False)
    save_df = save_df.drop(index=save_df.index)
    print("保存，行：" + str(index))


