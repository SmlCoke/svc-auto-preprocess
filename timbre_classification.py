import os
import shutil
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 =================
# 1. 你的“参考音频”文件夹 (存放确认是目标角色的干声，10-20个即可)
REF_DIR = Path("./yui")

# 2. 待筛选的“新切片”文件夹 (刚才脚本生成的 dataset_slices 目录)
# 注意：脚本会遍历这个目录下的所有子文件夹
INPUT_DIR = Path("./result/3-第3话-特训/")

# 3. 输出目录
OUTPUT_BASE = Path("./result_classification/")

# 4. 阈值设置 (分数范围通常在 0 ~ 1 之间)
# 大于此值：直接保留 (High Confidence)
THRESH_KEEP = 0.4  
# 小于此值但大于垃圾阈值：人工复核 (Unsure)
THRESH_REVIEW = 0.20
# 小于 0.20：直接认为是垃圾/杂音/其他人

# ===========================================

def load_model():
    print("正在加载 ECAPA-TDNN 模型...")
    # 自动使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 从 HuggingFace 加载预训练模型
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="tmp_model",
        run_opts={"device": device}
    )
    return verification

def get_embedding(model, wav_path):
    """提取单个音频的声纹向量"""
    signal, fs = torchaudio.load(wav_path)
    
    # === 修复：强制转为单声道 ===
    # signal 的形状通常是 [Channel, Time]
    # 如果通道数 > 1 (比如立体声), 对通道维度求平均
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    # ===========================

    # ECAPA-TDNN 需要 16k 采样率
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000).to(signal.device) # 确保设备一致
        signal = resampler(signal)
    
    # 提取 Embedding
    # 此时 signal 形状一定是 [1, N]，输出一定是 [1, 1, 192]
    embedding = model.encode_batch(signal)
    return embedding

def compute_reference_embedding(model, ref_dir):
    """计算参考音频的平均声纹向量"""
    print(f"正在分析参考音频: {ref_dir}")
    embeddings = []
    files = list(ref_dir.glob("*.wav")) + list(ref_dir.glob("*.flac"))
    
    if not files:
        raise FileNotFoundError("参考音频文件夹为空！请上传一些高质量干声。")

    for f in tqdm(files):
        try:
            emb = get_embedding(model, f)
            embeddings.append(emb)
        except Exception as e:
            print(f"跳过坏文件 {f.name}: {e}")
    
    if not embeddings:
        raise RuntimeError("无法提取任何参考特征")

    # 计算平均向量 (Centroid)
    # stack 形状: [N, 1, 192] -> mean -> [1, 1, 192]
    mean_embedding = torch.stack(embeddings).mean(dim=0)
    return mean_embedding

def main():
    # 准备输出目录
    dir_keep = OUTPUT_BASE / "1_Keep"
    dir_review = OUTPUT_BASE / "2_Review"
    dir_trash = OUTPUT_BASE / "3_Trash"
    
    for d in [dir_keep, dir_review, dir_trash]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    verification = load_model()

    # 2. 获取目标角色的声纹指纹
    target_emb = compute_reference_embedding(verification, REF_DIR)
    print("参考声纹提取完毕。开始大规模筛选...")

    # 3. 遍历待筛选文件
    # 使用 rglob 递归查找所有子文件夹里的 wav
    candidates = list(INPUT_DIR.rglob("*.wav"))
    print(f"共找到 {len(candidates)} 个待筛选切片。")

    for wav_file in tqdm(candidates):
        try:
            # 提取当前切片的声纹
            # 注意：太短的音频（<0.5s）可能会报错或极其不准
            info = torchaudio.info(str(wav_file))
            duration = info.num_frames / info.sample_rate
            if duration < 0.5:
                # 太短的直接扔进 Review 或 Trash
                shutil.copy2(str(wav_file), str(dir_trash / wav_file.name))
                continue

            current_emb = get_embedding(verification, wav_file)

            # 计算相似度 (Cosine Similarity)
            # score 是一个 tensor，取 .item() 拿数值
            score = torch.nn.functional.cosine_similarity(target_emb, current_emb, dim=-1).mean().item()

            # 决策移动
            if score >= THRESH_KEEP:
                dest = dir_keep
            elif score >= THRESH_REVIEW:
                dest = dir_review
            else:
                dest = dir_trash
            
            # 复制文件 (保留原始文件名，如果重名则自动处理)
            shutil.copy2(str(wav_file), str(dest / wav_file.name))
            
            # (可选) 打印调试信息，看分数分布
            # print(f"{wav_file.name}: {score:.4f} -> {dest.name}")

        except Exception as e:
            print(f"处理出错 {wav_file.name}: {e}")
            # 出错的文件扔进 Review
            shutil.copy2(str(wav_file), str(dir_review / wav_file.name))

    print("-" * 50)
    print("筛选完成！统计结果：")
    print(f"保留 (Keep): {len(list(dir_keep.glob('*.wav')))}")
    print(f"复核 (Review): {len(list(dir_review.glob('*.wav')))}")
    print(f"丢弃 (Trash): {len(list(dir_trash.glob('*.wav')))}")
    print(f"结果保存在: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()