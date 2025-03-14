# app服务
APP_BIND_ADDRESS = "0.0.0.0"
APP_PORT = 6060
WORKERS = 4  # 工作进程数
TIMEOUT = 600  # 超时时间
BACKLOG = 2048  # 当服务满载时，服务任务队列的最大长度
GPU_ID = 0

# 模型配置
PRETRAINED_MODEL_SAVE_DIR = 'model'
TARGET_SPEAKER_AUDIO_FILE_DIR = 'results/target_speaker_audio'
AUDIO_INPUT_FILE_DIR = 'results/meeting_audio'
ASR_RESULTS_JOSN_DIR = 'results/asr_result'
VERIFY_THRESHOLD = 0.6

# 测试数据
ASR_AUDIO = '20240716.wav'
ASR_RESULTS = 'asr_20240716.json'
TARGET_SPEAKER_AUDIO = '20240716_speaker_2.wav'

