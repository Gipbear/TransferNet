import os
import time
import requests

def _sparql_query(sparql_query):
    """执行 SPARQL 查询，返回 bindings 列表；HTTP 429 返回 None，其他错误返回 []"""
    url = 'https://query.wikidata.org/sparql'
    headers = {
        'User-Agent': 'WebQSP_MID_Mapper/1.0',
        'Accept': 'application/sparql-results+json'
    }
    try:
        response = requests.get(url, params={'query': sparql_query}, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()['results']['bindings']
        elif response.status_code == 429:
            print("错误: 请求过于频繁 (HTTP 429)。")
            return None
        else:
            print(f"查询失败，HTTP 状态码: {response.status_code}")
            return []
    except Exception as e:
        print(f"请求发生异常: {e}")
        return []


def batch_convert_mids_to_names(mid_list):
    """
    通过 Wikidata API 将 MIDs 批量转换为实体名称。
    - m. 前缀：使用 Freebase ID 属性 P646（格式 /m/xxx）
    - g. 前缀：使用 Google Knowledge Graph ID 属性 P2671（格式 /g/xxx）
    返回 {mid: name}，HTTP 429 返回 None。
    """
    m_mids = [mid for mid in mid_list if mid.strip().startswith('m.')]
    g_mids = [mid for mid in mid_list if mid.strip().startswith('g.')]
    result_mapping = {}

    # 查询 m. 前缀（P646）
    if m_mids:
        values_str = " ".join(f'"/m/{mid.strip()[2:]}"' for mid in m_mids)
        query = f"""
        SELECT ?mid ?itemLabel WHERE {{
          VALUES ?mid {{ {values_str} }}
          ?item wdt:P646 ?mid .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        bindings = _sparql_query(query)
        if bindings is None:
            return None
        for item in bindings:
            mid_val = item['mid']['value']
            if mid_val.startswith('/m/'):
                mid_val = 'm.' + mid_val[3:]
            label_val = item['itemLabel']['value']
            if not (label_val.startswith('Q') and label_val[1:].isdigit()):
                result_mapping[mid_val] = label_val

    # 查询 g. 前缀（P2671）
    if g_mids:
        values_str = " ".join(f'"/g/{mid.strip()[2:]}"' for mid in g_mids)
        query = f"""
        SELECT ?gkgId ?itemLabel WHERE {{
          VALUES ?gkgId {{ {values_str} }}
          ?item wdt:P2671 ?gkgId .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        bindings = _sparql_query(query)
        if bindings is None:
            return None
        for item in bindings:
            gkg_val = item['gkgId']['value']
            if gkg_val.startswith('/g/'):
                gkg_val = 'g.' + gkg_val[3:]
            label_val = item['itemLabel']['value']
            if not (label_val.startswith('Q') and label_val[1:].isdigit()):
                result_mapping[gkg_val] = label_val

    return result_mapping

def main():
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    fbwq_full_dir = os.path.join(root_dir, 'data', 'resources', 'WebQSP', 'fbwq_full')
    
    input_file = os.path.join(fbwq_full_dir, 'mid_entities.dict')
    mapped_file = os.path.join(fbwq_full_dir, 'mapped_entities.txt')
    unmapped_file = os.path.join(fbwq_full_dir, 'unmapped_entities.txt')

    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在。请先提取 mid_entities.dict。")
        return

    # 读取全部 mid（新格式：每行一个 MID，无数字 ID）
    print("正在加载所有需要查询的 MIDs...")
    all_mids = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            mid = line.strip()
            if mid.startswith('m.') or mid.startswith('g.'):
                all_mids.append(mid)
                
    # 读取已处理的 mid_list，实现断点续传
    processed_mids = set()
    if os.path.exists(mapped_file):
        with open(mapped_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    processed_mids.add(parts[0].strip())
    
    if os.path.exists(unmapped_file):
        with open(unmapped_file, 'r', encoding='utf-8') as f:
            for line in f:
                mid = line.strip()
                if mid:
                    processed_mids.add(mid)

    mids_to_process = [mid for mid in all_mids if mid not in processed_mids]
    
    print(f"总计实体数: {len(all_mids)}")
    print(f"已处理数: {len(processed_mids)}")
    print(f"剩余待处理数: {len(mids_to_process)}")
    
    if not mids_to_process:
        print("所有实体已处理完毕！")
        return
        
    # 查询配置
    BATCH_SIZE = 200
    # 限制总处理批次，以防死循环或一次性跑太久无法响应。若想跑全体可以将 max_batches 设为很大 (如 float('inf'))
    MAX_BATCHES_TEST = float('inf') 
    
    print(f"开始批量查询映射。每次处理 {BATCH_SIZE} 个。")
    
    mapped_f = open(mapped_file, 'a', encoding='utf-8')
    unmapped_f = open(unmapped_file, 'a', encoding='utf-8')
    
    batches_run = 0
    try:
        for i in range(0, len(mids_to_process), BATCH_SIZE):
            if batches_run >= MAX_BATCHES_TEST:
                print(f"已达到设定的测试批次上限 ({MAX_BATCHES_TEST})，停止执行。")
                break
                
            batch = mids_to_process[i:i+BATCH_SIZE]
            print(f"处理批次 {batches_run+1} : 索引 [{i} ~ {i+len(batch)-1}] ...", end=" ", flush=True)
            
            mapping = batch_convert_mids_to_names(batch)
            
            if mapping is None:
                print("查询网络失败或被限制，等待 10 秒后重试此批次...")
                time.sleep(30)
                # 简单重试一次
                mapping = batch_convert_mids_to_names(batch)
                
            if mapping is None:
                print("重试仍然失败，保存进度并退出。")
                break
                
            # 保存结果
            mapped_count = 0
            for mid in batch:
                if mid in mapping:
                    mapped_f.write(f"{mid}\t{mapping[mid]}\n")
                    mapped_count += 1
                else:
                    unmapped_f.write(f"{mid}\n")
            
            # 确保实时写入硬盘
            mapped_f.flush()
            unmapped_f.flush()
            
            print(f"成功. 本批找到映射: {mapped_count} / {len(batch)}。")
            
            batches_run += 1
            # 增加礼貌延迟，避免给 Wikidata 带来过大压力
            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\n检测到用户中断。正在安全退出...")
    except Exception as e:
        print(f"\n未知错误导致中断: {e}")
    finally:
        mapped_f.close()
        unmapped_f.close()
        print("文件句柄已关闭，进度已保存。")

if __name__ == '__main__':
    main()
