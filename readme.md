# Audio Slicing Pipeline

自动完成 UVR5(MDX + VR) - Slicer 的流程，支持批量处理

## 使用方法

1. 安装 python 3.10 版本
   ```bash
   conda create -n audio python=3.10
   conda activate audio
   ```
2. 安装 UVR5 的命令行版 (audio-separator)
    ```bash
    pip install "audio-separator[gpu]"
    ```
3. 安装 `ffmpeg`
   ```bash
   conda install -y -c conda-forge ffmpeg
   ```
4. 安装其他依赖库
    ```bash
    pip install -r requirements.txt
    ```
5. 下载 MDX 和 VR 模型文件，放在根目录即可
   ```bash
   # MDX 模型
   wget -P ./ "https://huggingface.co/Eddycrack864/audio-separator-models/resolve/main/mdx-net/UVR-MDX-NET-Inst_HQ_5.onnx"
   # VR 模型
   wget -P ./ "https://huggingface.co/Eddycrack864/audio-separator-models/resolve/main/vr-arch/5_HP-Karaoke-UVR.pth"
   ```
6. 然后运行即可
   ```bash
   python auto_process.py
   ```

## 设备环境
- 推荐使用 RTX 4090 或更高的显卡，VR 模型在 RTX 5090 上表现更佳