import json
from typing import Dict, Any, List

def load_json(file_path: str) -> Any:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_speaker(asr_entry: Dict, test_in_data: Dict) -> int:
    """Find the speaker ID for an ASR entry based on time overlap."""
    # Convert ASR times from milliseconds to seconds
    asr_start = asr_entry["start_time"] / 1000.0
    asr_end = asr_entry["end_time"] / 1000.0
    
    best_match_speaker = None
    best_overlap = 0
    
    for segment_key, segment_data in test_in_data.items():
        seg_start = segment_data["start"]
        seg_end = segment_data["stop"]
        
        # Check if there's an overlap
        if max(asr_start, seg_start) < min(asr_end, seg_end):
            overlap = min(asr_end, seg_end) - max(asr_start, seg_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_speaker = segment_data["speaker"]
    
    return best_match_speaker

def save_results_to_json(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_file}")

def extract_asr_with_speakers(asr_file: str,
                             test_in_file: str, 
                             target_speaker_id: int = None,
                             print_to_console:bool=False,
                             output_json:str=None):
    """Print ASR text with speaker information."""
    # Load data
    asr_data = load_json(asr_file)
    test_in_data = load_json(test_in_file)
    
    # 临时存储，用于合并相同speaker的文本
    merged_results = []

    current_speaker = None
    current_text = ""
    
    # ANSI color codes
    RED = "\033[31m"
    RESET = "\033[0m"
    print(f"\033[32m###################")
    for asr_entry in asr_data:
        speaker_id = find_speaker(asr_entry, test_in_data)
        
        # 如果是新的speaker或者是第一条记录
        if speaker_id != current_speaker:
            # 保存之前的记录(如果存在)
            if current_text:
                merged_results.append({
                    "speaker": "myself" if current_speaker == target_speaker_id else current_speaker,
                    "text": current_text
                })
            
            # 开始新的记录
            current_speaker = speaker_id
            current_text = asr_entry['content']
        else:
            # 合并相同speaker的文本
            current_text += " " + asr_entry['content']
    
    # 添加最后一条记录
    if current_text:
        merged_results.append({
            "speaker": "myself" if current_speaker == target_speaker_id else current_speaker,
            "text": current_text
        })
    
    # 如果需要打印结果
    if print_to_console:
        for item in merged_results:
            if item["speaker"] == "myself":
                speaker_display = "me"
                # 使用红色打印目标说话人的内容
                print(f"{RED}{speaker_display}: {item['text']}{RESET}")
            else:
                speaker_display = f"Speaker {item['speaker']}"
                print(f"{speaker_display}: {item['text']}")
    
    # 如果指定了输出文件名，则保存为JSON
    if output_json:
        save_results_to_json(merged_results, output_json)
    
    return merged_results

if __name__ == "__main__":
    diarization_file = "./file_save_dir/meeting_audio/sd_result_20240716.json"
    asr_file = "./file_save_dir/asr_result/asr_20240716.json"
    output_json = "./file_save_dir/extracted_results.json"
    
    # 指定目标说话人ID为2，将对应说话人标记为"me"
    target_speaker = 2
    
    results = extract_asr_with_speakers(
        asr_file, 
        diarization_file, 
        target_speaker, 
        print_to_console=True,
        output_json=output_json
    )