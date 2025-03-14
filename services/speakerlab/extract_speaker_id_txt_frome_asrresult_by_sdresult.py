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

def merge_speaker_texts(messages):
    result = []
    current_speaker = None
    current_text = []
    
    for msg in messages:
        if msg['speaker'] != current_speaker:
            # Save previous speaker's merged text
            if current_speaker is not None:
                result.append({
                    'speaker': current_speaker,
                    'text': ''.join(current_text)
                })
            # Start new speaker
            current_speaker = msg['speaker']
            current_text = [msg['text']]
        else:
            # Add to current speaker's text
            current_text.append(msg['text'])
    
    # Add the last speaker's text
    if current_text:
        result.append({
            'speaker': current_speaker,
            'text': ''.join(current_text)
        })
        
    return result

def extract_asr_with_speakers(asr_file: str,
                               test_in_data: list, 
                               target_speaker_id: int = None,
                              ):
    """Print ASR text with timestamps and speaker information."""
    # Load data
    asr_data = load_json(asr_file)
    # test_in_data = test_in_file
    
    # Prepare output
    output_lines = []
    results_list = []
    
    for asr_entry in asr_data:
        speaker_id = find_speaker(asr_entry, test_in_data)
        
        # Determine speaker label
        speaker_label = "me" if speaker_id == target_speaker_id else f"Speaker {speaker_id}"
        
        # Create output line for console printing
        line = f"{speaker_label}: {asr_entry['content']}"
        output_lines.append(line)
        
        # Create dictionary for results list
        results_list.append({
            "speaker": "myself" if speaker_id == target_speaker_id else speaker_id,
            "text": asr_entry['content']
        })
    
    results = merge_speaker_texts(results_list)
    return results