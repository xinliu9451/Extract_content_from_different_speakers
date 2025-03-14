from fastapi import FastAPI
from pydantic import BaseModel
import time
import sys
import os

# from services.diarization_service import speaker_diarization_pipeline
from services.speakerlab.infer_sd_service import SpeakerDiarizationService
from utils.log import logger
import config

app = FastAPI()

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_dir = os.path.dirname(current_dir)

# 获取 services 目录的路径
services_dir = os.path.join(parent_dir, 'services')

# 将 services 目录添加到 sys.path
if services_dir not in sys.path:
    sys.path.append(services_dir)


# 加载模型
speaker_diarization_service = SpeakerDiarizationService(
        pretrained_model_save_dir=config.PRETRAINED_MODEL_SAVE_DIR,
        GPU_id=config.GPU_ID   
    )

# 定义请求数据模型
class AudioRequest(BaseModel):
    audio_url: str
    audio_results: str
    speaker_id: str

@app.post("/get_diarization_results")
async def get_diarization_results(request: AudioRequest):

    logger.info(f"speaker_id={request.speaker_id} | func=get_diarization_results | Successfully accessed the service")  # 通过日志记录筛选成功访问的请求，可以分析访问量
    start_time = time.time()
    # 说话人分离
    try:
        sd_result_json = speaker_diarization_service.speaker_diarization_pipline(request.audio_url,  # asr 音频文件
                                                                                    include_overlap=False,
                                                                                    speaker_num=None  # 可以选择说话人数量，如果指定，则按照指定数量进行说话人分离
                                                                                    )
        logger.info(f"speaker_id={request.speaker_id} | func=speaker_diarization_pipline | Successfully performed speaker separation")
    except Exception as e:
        logger.error(f"speaker_id={request.speaker_id} | func=speaker_diarization_pipline | Failed to perform speaker separation")
        return {"error": "Failed to perform speaker separation"}

    # 识别用户说话人id
    try:
        target_speaker_id = speaker_diarization_service.speaker_verification_pipline(request.audio_url,
                                                                                    sd_result_json,
                                                                                    request.speaker_id,  # 目标说话人音频文件
                                                                                    verify_threshold=config.VERIFY_THRESHOLD)
        logger.info(f"speaker_id={request.speaker_id} | func=speaker_verification_pipline | Successfully identified user ID")
    except Exception as e:
        logger.error(f"speaker_id={request.speaker_id} | func=speaker_verification_pipline | Failed to identify user ID")
        return {"error": "Failed to identify user ID"}

    # 提取asr结果
    try:
        results = speaker_diarization_service.extract_asr_results(request.audio_results, # asr结果json文件
                                                                        sd_result_json,
                                                                        target_speaker_id
                                                                        )
        logger.info(f"speaker_id={request.speaker_id} | func=extract_asr_results | Successfully extracted user's speech content")
    except Exception as e:
        logger.error(f"speaker_id={request.speaker_id} | func=extract_asr_results | Failed to extract user's speech content")
        return {"error": "Failed to extract user's speech content"}

    print('总耗时：', time.time() - start_time)

    return results