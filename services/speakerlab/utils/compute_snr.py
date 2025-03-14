import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def calculate_snr(audio_path):
    """
    计算语音信噪比 (SNR)
    
    参数:
        audio_path: 音频文件路径
        noise_start: 噪声段开始时间（秒）
        noise_end: 噪声段结束时间（秒），如果为None则使用signal_start
        signal_start: 信号段开始时间（秒），如果为None则使用noise_end
        signal_end: 信号段结束时间（秒），如果为None则使用整个文件长度
    
    返回:
        snr_db: 信噪比（分贝）
    """
    # 加载音频文件
    y, sr = sf.read(audio_path)

    # 假设前5%是噪声，后95%是信号
    noise_start = 0
    noise_end = int(len(y) * 0.05)
    signal_start = noise_end
    signal_end = len(y)
    
    # 提取噪声和信号段
    noise = y[noise_start:noise_end]
    signal = y[signal_start:signal_end]
    
    # 计算信号功率
    signal_power = np.mean(signal**2)
    
    # 计算噪声功率
    noise_power = np.mean(noise**2)
    
    # 计算信噪比（分贝）
    if noise_power > 0:
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)
    else:
        snr_db = float('inf')  # 如果噪声功率为0，SNR为无穷大
    
    return snr_db

# 使用示例
if __name__ == "__main__":
    # 替换为你的音频文件路径
    audio_file = "../test_logs/test4compute_snr.wav"
    
    # 计算SNR
    snr = calculate_snr(audio_file)
    print(f"信噪比 (SNR): {snr:.2f} dB")