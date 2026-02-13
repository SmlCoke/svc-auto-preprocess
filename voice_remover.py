import os
import shutil
import subprocess
import librosa
import soundfile as sf
import logging
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 输入音频所在的文件夹
INPUT_DIR = Path("D:\\Project\\svc\\svc-data\\K-ON\\Main-Vocal\\K-ON-ONE\\dataset_slices\\2-第2话-乐器")
# 输出结果的文件夹
OUTPUT_DIR = Path("D:\\Project\\svc\\svc-data\\K-ON\\Main-Vocal\\K-ON-ONE\\dataset_slices\\2-第2话-乐器-new")
# 临时文件夹
TEMP_DIR = Path("D:\\Project\\svc\\svc-data\\K-ON\\Main-Vocal\\K-ON-ONE\\temp")

# --- 模型路径 ---
# 第一步：MDX 模型 (去主要伴奏)
MODEL_PATH_MDX = Path("D:\\Project\\svc\\software\\Ultimate-Vocal-Remover\\Ultimate Vocal Remover\\models\\MDX_Net_Models\\UVR-MDX-NET-Inst_HQ_5.onnx")
# 第二步：VR 模型 (去残留和声/杂音)
MODEL_PATH_VR = Path("D:\\Project\\svc\\software\\Ultimate-Vocal-Remover\\Ultimate Vocal Remover\\models\\VR_Models\\5_HP-Karaoke-UVR.pth")

# 是否采用强力模式（再执行一轮 VR + MDX）
Power = False 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_filename(original_name):
    stem = original_name.rsplit('.', 1)[0]
    stem = stem.replace("！-1080P 高码率-AVC", "")
    stem = stem.replace(" ", "-")
    stem = stem.strip("-")
    return stem

def convert_to_wav(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", "44100",
        "-ac", "2",
        "-loglevel", "error",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def run_separator(input_wav, output_dir, model_path):
    """
    通用的分离函数，支持 MDX 和 VR 模型
    """
    model_dir = model_path.parent
    model_name = model_path.name
    
    # === MDX 模型兼容性补丁 (针对 HQ_5) ===
    if "HQ_5" in model_name and model_path.suffix == ".onnx":
        compat_name = "UVR-MDX-NET-Inst_HQ_3.onnx"
        compat_path = model_dir / compat_name
        if not compat_path.exists():
            logging.info("应用 HQ_5 -> HQ_3 兼容性软链接...")
            os.symlink(model_path, compat_path)
        model_name = compat_name

    cmd = [
        "audio-separator", str(input_wav),
        "--model_filename", model_name,
        "--model_file_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--output_format", "wav",
        "--single_stem", "Vocals",
        "--normalization", "0.9"
    ]
    
    # 如果是 VR 模型 (.pth)，我们可以加一些特定参数 (可选)
    # --vr_window_size 320 (更快) 或 512 (质量更好，默认)
    # 这里保持默认即可，RTX 5090 跑 512 也是秒杀
    
    logging.info(f"正在执行分离 (模型: {model_name}) ...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"UVR5 处理出错: {result.stderr}")
        raise RuntimeError(f"Separator failed on {model_name}")
    
    # 查找最新的输出文件
    # audio-separator 输出的文件名会包含模型名，例如 file_(Vocals)_Model.wav
    # 为了防止找到旧文件，我们在外部控制 temp 目录的清理
    files = list(output_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError("未找到分离后的文件")
    
    # 返回最新的文件
    return max(files, key=os.path.getctime)

def main():
    if not INPUT_DIR.exists():
        logging.error(f"输入目录不存在: {INPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH_MDX.exists() or not MODEL_PATH_VR.exists():
        logging.error(f"模型文件缺失，请检查路径:\nMDX: {MODEL_PATH_MDX}\nVR: {MODEL_PATH_VR}")
        return

    audio_files = sorted([f for f in INPUT_DIR.iterdir() if f.suffix.lower() == ".wav"])
    
    print(f"检测到 {len(audio_files)} 个文件，开始两阶段处理 (MDX -> VR)...")
    print("-" * 50)

    for audio_file in tqdm(audio_files):
        try:
            parent_folder_name = audio_file.parent.name
            final_dir = OUTPUT_DIR / parent_folder_name
            final_dir.mkdir(parents=True, exist_ok=True)

            new_name = clean_filename(audio_file.name)
            logging.info(f"处理任务: {new_name}")

            # --- 清理临时目录 ---
            for f in TEMP_DIR.iterdir():
                if f.is_file(): f.unlink()

            # --- 步骤 1: MDX 去伴奏 ---
            # 输出会包含 (Vocals)_MDX...
            step1_wav = run_separator(audio_file, TEMP_DIR, MODEL_PATH_MDX)

            # --- 步骤 2: VR 去杂音 (输入是步骤1的产物) ---
            step2_wav = run_separator(step1_wav, TEMP_DIR, MODEL_PATH_VR)

            final_wav = final_dir / f"{new_name}.wav"
            shutil.move(step2_wav, final_wav)

            if Power:
                # 强力模式：再执行一轮 VR + MDX
                logging.info(f"批准执行强力模式 (再一轮 VR + MDX) ...")
                step3_wav = run_separator(step2_wav, TEMP_DIR, MODEL_PATH_MDX)
                step4_wav = run_separator(step3_wav, TEMP_DIR, MODEL_PATH_VR)
                shutil.move(step4_wav, final_wav)
                logging.info(f"完成 (强力模式): {final_wav}")
                step3_wav.unlink(missing_ok=True)
                step4_wav.unlink(missing_ok=True)

            
            logging.info(f"完成: {final_wav}")
            step1_wav.unlink(missing_ok=True)
            step2_wav.unlink(missing_ok=True)
           

        except Exception as e:
            logging.error(f"处理文件 {audio_file.name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

    # 最终清理
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    print("-" * 50)
    print(f"全部处理完成！结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()