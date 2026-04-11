"""
扫描 data/resources/WebQSP 目录下所有文件（含 entities.dict、KG 三元组、QA 文件），
收集全部 MID，写入 mid_entities.dict（格式：每行一个 MID，无数字 ID）。

用法：
    python -m WebQSP.build_mid_entities --input_dir data/resources/WebQSP
"""
import argparse
import os
import re


def collect_all_mids(input_dir):
    """从 entities.dict、KG 三元组和 QA 文件中收集所有 MID"""
    all_mids = set()

    # entities.dict：第一列为实体字符串，过滤出 m./g. 开头的 MID
    entities_path = os.path.join(input_dir, 'fbwq_full/entities.dict')
    if os.path.exists(entities_path):
        with open(entities_path) as f:
            for line in f:
                key = line.split('\t')[0].strip()
                if key.startswith('m.') or key.startswith('g.'):
                    all_mids.add(key)

    # KG 三元组文件：subject\trelation\tobject
    for fn in ['fbwq_full/train.txt', 'fbwq_full/test.txt', 'fbwq_full/valid.txt']:
        fpath = os.path.join(input_dir, fn)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    for col in (parts[0], parts[2]):
                        col = col.strip()
                        if col.startswith('m.') or col.startswith('g.'):
                            all_mids.add(col)

    # QA 文件：question [topic_mid]\tanswer_mid1|answer_mid2|...
    qa_files = [
        'QA_data/WebQuestionsSP/qa_train_webqsp.txt',
        'QA_data/WebQuestionsSP/qa_test_webqsp.txt',
        'QA_data/WebQuestionsSP/qa_test_webqsp_fixed.txt',
    ]
    for fn in qa_files:
        fpath = os.path.join(input_dir, fn)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                m = re.search(r'\[([^\]]+)\]', parts[0])
                if m:
                    mid = m.group(1)
                    if mid.startswith('m.') or mid.startswith('g.'):
                        all_mids.add(mid)
                for amid in parts[1].split('|'):
                    amid = amid.strip()
                    if amid.startswith('m.') or amid.startswith('g.'):
                        all_mids.add(amid)

    return all_mids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='data/resources/WebQSP 目录路径')
    args = parser.parse_args()

    out_path = os.path.join(args.input_dir, 'fbwq_full/mid_entities.dict')

    print('收集目录下所有 MID ...')
    all_mids = collect_all_mids(args.input_dir)
    print(f'  共发现 {len(all_mids)} 个唯一 MID')

    print(f'写入 {out_path} ...')
    with open(out_path, 'w') as f:
        for mid in sorted(all_mids):
            f.write(mid + '\n')

    print(f'完成，共写入 {len(all_mids)} 条。')


if __name__ == '__main__':
    main()
