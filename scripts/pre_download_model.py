from unsloth import FastLanguageModel
import torch
import time

# 1. 配置（与你一致）
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
max_seq_length = 2048

# 2. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 设置 Padding Token（Batch 推理必须）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 设置压测参数：通过增加 Batch Size 来拉满 GPU 利用率
# 如果你的显存很大（如 24G），可以将 batch_size 改为 16 或 32
batch_size = 8 
prompt = "请解释什么是量子纠缠。"
prompts = [prompt] * batch_size

# 4. 准备输入并移动到 CUDA
inputs = tokenizer(
    prompts, 
    return_tensors = "pt", 
    padding = True
).to("cuda")

# 5. 重置显存统计并开始推理
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats() # 重置峰值统计
start_mem = torch.cuda.memory_allocated() / 1024**3

print(f"--- 开始压测 ---")
print(f"Batch Size: {batch_size}")

start_time = time.time()

# 执行生成
outputs = model.generate(
    **inputs, 
    max_new_tokens = 128, 
    use_cache = True
)

end_time = time.time()

# 6. 性能计算
total_time = end_time - start_time
# 计算生成的 Token 总数（排除输入的 Prompt Tokens）
input_len = inputs.input_ids.shape[1]
output_len = outputs.shape[1]
new_tokens_per_sample = output_len - input_len
total_new_tokens = new_tokens_per_sample * batch_size

tokens_per_sec = total_new_tokens / total_time
peak_mem = torch.cuda.max_memory_reserved() / 1024**3 # 物理预留的最高显存

# 7. 打印结果
print("-" * 30)
print(f"推理总耗时: {total_time:.2f} 秒")
print(f"生成的 Token 总数: {total_new_tokens} ({batch_size} x {new_tokens_per_sample})")
print(f"平均吞吐量 (Throughput): {tokens_per_sec:.2f} tokens/s")
print(f"模型基础显存: {start_mem:.2f} GB")
print(f"运行峰值显存 (Reserved): {peak_mem:.2f} GB")
print("-" * 30)