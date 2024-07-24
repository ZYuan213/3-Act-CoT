import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine
import nltk

# model = SentenceTransformer('/root/ZYuan/huggingface_model/paraphrase-multilingual-mpnet-base-v2') #多语言
model = SentenceTransformer('all-mpnet-base-v2')  # 英语

path = ''
dataset = pd.read_csv(path + '') #Your candidate stories path

data = {
    'inputs': [],
    'selected_gen_story': [],
    'label': [],
    'cluster_label': [],
    'chosen_story': []
}
save_df = pd.DataFrame(data)
alpha = 0.5


def caculate_coherence_relevance_score(input_embedding, story_text):
    sentences = nltk.sent_tokenize(story_text)
    co_scores = []
    re_scores = []
    sentence_embedding = [model.encode(s) for s in sentences]
    for index in range(1, len(sentences)):
        if index == 1:
            re_scores.append(cosine(input_embedding, sentence_embedding[0]))
        re_scores.append(cosine(input_embedding, sentence_embedding[index]))
        co_scores.append(cosine(sentence_embedding[index - 1], sentence_embedding[index]))

    result = alpha * np.mean(co_scores) + (1 - alpha) * np.mean(re_scores)
    return result


COT_path_num = 20
n_clusters = 3
distance_threshold = None
for _, row in tqdm(dataset.iterrows()):
    story_embeddings = []
    story_texts = []
    for i in range(1, COT_path_num + 1):
        story_embeddings.append(model.encode(row[f'gen_story_{i}']).tolist())
        story_texts.append(row[f'gen_story_{i}'])

    AC = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', distance_threshold=distance_threshold).fit(
        story_embeddings)
    """
    选择具有最多样本的簇    
    """
    cluster_ids, cluster_sizes = np.unique(AC.labels_, return_counts=True)
    # 选择具有最多元素的簇
    most_label_index = np.argmax(cluster_sizes)
    most_label = cluster_ids[most_label_index]  # 包含样本最多的簇的标签

    label_points_indices = np.where(AC.labels_ == most_label)[0]
    input_embedding = model.encode(row['inputs'])
    co_re_scores = []
    for idx in label_points_indices:
        s = caculate_coherence_relevance_score(input_embedding=input_embedding, story_text=story_texts[idx])
        co_re_scores.append(s)

    max_idx = np.argmax(co_re_scores)
    target_index = label_points_indices[max_idx] + 1  # 与输入最接近的样本
    new_data = [{
        'inputs': row['inputs'],
        # 'inputs':model_input,
        'selected_gen_story': row[f'gen_story_{target_index}'],
        'label': row['label'],
        'cluster_label': AC.labels_,
        'chosen_story': target_index - 1
    }]

    new_data = pd.DataFrame(new_data)
    save_df = pd.concat([save_df, new_data], ignore_index=True)
save_df.to_csv(
    path + f'/selected_gen_0415_AgglomerativeClustering_{n_clusters}_{distance_threshold}_{COT_path_num}_re_co_{alpha}.csv',
    mode="w", encoding="utf_8_sig")