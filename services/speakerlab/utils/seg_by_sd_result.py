import os
import json
from pydub import AudioSegment

# 根据说话人分类结果切割音频并按说话人ID保存
def cut_audio_by_sd_result(audio_file, diarization_results_file, output_dir):
    """
    根据说话人分类结果切割原始音频并按说话人ID保存
    
    参数:
        audio_file: 原始音频文件路径
        diarization_results: 说话人分类结果字典
        output_dir: 输出目录
    """
    with open(diarization_results_file, 'r') as f:
        diarization_results = json.load(f)
   
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有说话人ID
    speaker_ids = set(segment_info["speaker"] for segment_info in diarization_results.values())
    
    # 为每个说话人创建单独的目录
    for speaker_id in speaker_ids:
        speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
        os.makedirs(speaker_dir, exist_ok=True)
    
    # 加载音频文件
    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        print(f"无法加载音频文件: {e}")
        return
    
    # 按照说话人分割音频
    for segment_name, segment_info in diarization_results.items():
        start_time = segment_info["start"]
        end_time = segment_info["stop"]
        speaker_id = segment_info["speaker"]
        
        # 转换为毫秒
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # 提取音频片段
        audio_segment = audio[start_ms:end_ms]
        
        # 保存音频片段
        output_file = os.path.join(output_dir, f"speaker_{speaker_id}", f"{segment_name}.wav")
        audio_segment.export(output_file, format="wav")
        
        print(f"已保存片段: {segment_name} (说话人 {speaker_id}, {start_time:.2f}s - {end_time:.2f}s)")
    
    print(f"音频已按说话人切割并保存到 {output_dir}")


# 执行音频切割
if __name__ == "__main__": 
    import os
    import json
    
    # 创建输出目录
    audio_file = "./file_save_dir/meeting_audio/real_record_meeting_2_woman_overlap.wav"
    diarization_results_file = "./file_save_dir/meeting_audio/real_record_meeting_2_woman_overlap.json"
    audio_segments_root_dir = "./file_save_dir/speaker_segmentation_result"
    # 切割音频并保存
    audio_segments_dir = os.path.join(audio_segments_root_dir, os.path.basename(audio_file).split('.')[0])
    cut_audio_by_sd_result(audio_file, diarization_results_file, audio_segments_dir)
    