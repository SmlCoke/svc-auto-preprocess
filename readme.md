# Audio Slicing Pipeline

自动完成 UVR5(MDX + VR) - Slicer 的流程，支持批量处理

## 使用方法

1. 安装 python 3.10 版本
   ```bash
   conda create -n audio python=3.10
   conda activate audio
   ```
2. 安装 `UVR5` 的命令行版 (audio-separator)
    ```bash
    pip install "audio-separator[gpu]"
    ```
3. 安装 `ffmpeg`
   ```bash
   conda install -y -c conda-forge ffmpeg
   ```
4. 安装 `speechbrain` 包用于自动识别对应角色的切片并分类
   ```bash
   pip install speechbrain
   ```
5. 安装其他依赖库
    ```bash
    pip install -r requirements.txt
    ```
6. 下载 MDX 和 VR 模型文件，放在目录 `./uvr_model/` 下
   ```bash
   # MDX 模型
   wget -P ./uvr_model/ "https://huggingface.co/Eddycrack864/audio-separator-models/resolve/main/mdx-net/UVR-MDX-NET-Inst_HQ_5.onnx"
   # VR 模型
   wget -P ./uvr_model/ "https://huggingface.co/Eddycrack864/audio-separator-models/resolve/main/vr-arch/5_HP-Karaoke-UVR.pth"
   ```
7. 对**动漫剧集整体去除杂音，并切片**
   ```bash
   python auto_process.py
   ```
   去除杂音的流程是：MDX-NET-Inst_HQ_5模型 $rightarrow$ 5_HP-Karaoke-UVR.pth模型
   该脚本中可以设置参数：`Power = True` 进行强去除（再接一次 VR + MDX），但可能会损失一些人声细节，建议先使用默认的 `Power = False` 进行处理，如果效果不理想再尝试 `Power = True`。
8. 对**单份切片去除杂音**
   ```bash
   python voice_process.py
   ```
   该脚本中也可以选择是否进行强去除，方法同上。
9. 对**切片进行角色识别**
   ```bash
   python timbre_classification.py
   ```
   注意：该脚本运行时会动态下载预训练模型文件，首次运行时请耐心等待。
   此外，为了进行角色识别，需要你提供该角色 10~20 份**纯人声样本**，放到 `REF_DIR` 目录下，推荐`.wav`/`.flac`。
   该脚本会自动识别 `INPUT_DIR` 目录下哪些切片可能包含该角色的声音，并将其复制到 `OUTPUT_DIR` 目录下，供后续使用。
   结果分类三类：`KEEP`（包含该角色声音的切片），`REVIEW`（可能包含该角色声音的切片，需要人工复核），`Trash`（不包含该角色声音的切片）。你可以根据实际情况调整分类的阈值，参数分别为 `THRESH_KEEP` 和 `THRESH_REVIEW`
 
## 设备环境
- 推荐使用 RTX 4090 或更高的显卡，VR 模型在 RTX 5090 上表现更佳