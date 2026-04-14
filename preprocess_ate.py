# 文件名: S_MA/preprocess_ate.py
import spacy
import json
from tqdm import tqdm
import os

OPINION_POS_TAGS = {'ADJ', 'ADV', 'VERB'}
# 定义用于检查句法关联的最大依赖距离
MAX_DEP_DISTANCE = 2


def get_dep_distance(token1, token2):
    """
    计算两个token在依存树中的最短距离。
    如果距离过长或路径复杂，返回一个较大的值。
    对于预处理的启发式方法，我们只检查直接或间接的父子关系。
    """
    if token1 == token2:
        return 0

    # 1. 检查直接的父子关系
    if token1.head == token2 or token2.head == token1:
        return 1

    # 2. 检查间接的父子关系（距离为2）
    # token1 -> head1 -> token2
    if token1.head.head == token2:
        return 2
    # token2 -> head2 -> token1
    if token2.head.head == token1:
        return 2

    # 3. 检查共同的直接父节点
    if token1.head == token2.head:
        return 2

    return MAX_DEP_DISTANCE + 1  # 超过阈值


def extract_aspect_words(text, nlp):
    """
    使用 spacy 的名词短语 (noun_chunks) 作为方面词的提取方法，
    并结合句法依赖关系和词性信息进行过滤 (借鉴论文的多视角语言特征思想)。
    """
    if not isinstance(text, str):
        print(f"警告：收到了非字符串类型 {type(text)}，返回空列表。")
        return []

    doc = nlp(text)

    # 1. 提取所有名词短语作为候选方面词
    candidate_aspects = [chunk for chunk in doc.noun_chunks]

    # 2. 提取所有潜在观点词
    opinion_tokens = [token for token in doc if token.pos_ in OPINION_POS_TAGS]

    final_aspects = []

    # 3. 基于句法关联进行过滤
    for chunk in candidate_aspects:
        is_relevant = False

        for aspect_token in chunk:
            for opinion_token in opinion_tokens:

                if get_dep_distance(aspect_token, opinion_token) <= MAX_DEP_DISTANCE:
                    is_relevant = True
                    break

            if is_relevant:
                break

        if is_relevant:
            final_aspects.append(chunk.text)

    return list(set(final_aspects))


def process_file(input_path, output_path, nlp):
    """
    读取、处理并写入新的 .json 文件
    """
    if not os.path.exists(input_path):
        print(f"警告：找不到文件 {input_path}，跳过。")
        return False

    print(f"正在处理 {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：读取 {input_path} 时发生 JSON 解码错误: {e}。跳过此文件。")
        return False

    processed_list = []
    if not isinstance(data_list, list):
        print(f"错误：{input_path} 中的内容不是一个 JSON 列表。跳过此文件。")
        return False

    for data in tqdm(data_list, desc="  提取方面词"):
        if not isinstance(data, dict):
            print(f"警告：在 {input_path} 中发现非字典条目 {type(data)}，跳过。")
            continue

        text = data.get('text', '')

        aspect_words = extract_aspect_words(text, nlp)
        data['aspect_words'] = aspect_words

        processed_list.append(data)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_list, f, indent=4)
        print(f"处理完成，已保存到 {output_path}")
        return True
    except IOError as e:
        print(f"错误：无法写入到 {output_path}: {e}")
        return False


if __name__ == '__main__':
    # 加载 spacy 模型
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("错误：无法加载 'en_core_web_sm'。")
        print("请确保您已在终端运行过: python -m spacy download en_core_web_sm")
        exit()

    print("开始预处理数据集以生成方面词...")

    # 定义所有数据集的基础路径
    # 这些路径是根据 S_MA 文件夹内的相对路径
    base_data_dir = ''

    # 1. 处理 HFM 数据集
    print("\n--- 正在处理 HFM ---")
    hfm_path_root = os.path.join(base_data_dir, 'HFM')
    hfm_files = ['train.json', 'valid.json', 'test.json']
    for file_name in hfm_files:
        in_path = os.path.join(hfm_path_root, file_name)
        out_path = os.path.join(hfm_path_root, file_name.replace('.json', '_processed.json'))
        process_file(in_path, out_path, nlp)

    # 2. 处理 MVSA-single 数据集
    print("\n--- 正在处理 MVSA-single (10-flod-1) ---")
    mvsa_s_path_root = os.path.join(base_data_dir, 'MVSA-single', '10-flod-1')
    mvsa_s_files = ['train.json', 'dev.json', 'test.json']
    for file_name in mvsa_s_files:
        in_path = os.path.join(mvsa_s_path_root, file_name)
        out_path = os.path.join(mvsa_s_path_root, file_name.replace('.json', '_processed.json'))
        process_file(in_path, out_path, nlp)

    # 3. 处理 MVSA-multiple 数据集
    print("\n--- 正在处理 MVSA-multiple (10-flod-1) ---")
    mvsa_m_path_root = os.path.join(base_data_dir, 'MVSA-multiple', '10-flod-1')
    mvsa_m_files = ['train.json', 'dev.json', 'test.json']
    for file_name in mvsa_m_files:
        in_path = os.path.join(mvsa_m_path_root, file_name)
        out_path = os.path.join(mvsa_m_path_root, file_name.replace('.json', '_processed.json'))
        process_file(in_path, out_path, nlp)

    print("\n所有数据集预处理已完成。")
