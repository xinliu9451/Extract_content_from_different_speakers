import requests
import json
import os

import config

# API地址
url = f"http://192.168.110.49:{config.APP_PORT}/get_diarization_results"

# 请求数据
data = {
    "audio_url": os.path.join(config.AUDIO_INPUT_FILE_DIR, config.ASR_AUDIO),
    "audio_results": os.path.join(config.ASR_RESULTS_JOSN_DIR, config.ASR_RESULTS),
    "speaker_id": os.path.join(config.TARGET_SPEAKER_AUDIO_FILE_DIR, config.TARGET_SPEAKER_AUDIO)
}

# 发送POST请求
try:
    response = requests.post(
        url=url,
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    # 检查响应状态
    if response.status_code == 200:
        print("请求成功！")
        print("响应数据：", response.json())
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print("错误信息：", response.text)

except requests.exceptions.RequestException as e:
    print(f"发生错误：{e}")
