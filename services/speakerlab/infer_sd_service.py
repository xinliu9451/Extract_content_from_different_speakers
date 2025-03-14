import os
import json
import warnings
import torch
import numpy as np
from torch.nn import functional as F
from services.speakerlab.utils.fileio import load_audio
from services.speakerlab.infer_diarization import Diarization3Dspeaker
from services.speakerlab.get_pretrained_model import get_voice_activity_detection_model, get_segmentation_model_from_ckpt, get_speaker_embedding_model
from services.speakerlab.extract_speaker_id_txt_frome_asrresult_by_sdresult import extract_asr_with_speakers
from services.speakerlab.utils.fileio import load_json_file

warnings.filterwarnings("ignore")
 

class SpeakerDiarizationService:
    def __init__(self,pretrained_model_save_dir=None,GPU_id=None):
        self.pretrained_model_save_dir = pretrained_model_save_dir
        self.device = torch.device('cuda:%d'%(GPU_id))

        # 1. 初始化需要用到的几个模型
        self.vad_model = get_voice_activity_detection_model(self.device, self.pretrained_model_save_dir)
        self.embedding_model, self.feature_extractor = get_speaker_embedding_model(self.device, self.pretrained_model_save_dir)
        self.segmentation_model = get_segmentation_model_from_ckpt(self.device, self.pretrained_model_save_dir)

        self.fs = self.feature_extractor.sample_rate
        # 2. 初始化speaker_diarization流程
        self.speaker_diarization = Diarization3Dspeaker(
            self.vad_model,
            self.embedding_model,
            self.feature_extractor,
            self.segmentation_model,
            self.device
        )
    @staticmethod
    def normalize(audio, target_level=-25):
            # normalize the signal to the target
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms + 1e-10)
            return audio * scalar
    
    def compute_embedding(self, audio_data):
        audio_data = self.normalize(audio_data)
        # compute feat
        feat = self.feature_extractor(audio_data).unsqueeze(0).to(self.device) # torch
        # compute embedding
        with torch.no_grad():
            embedding = self.embedding_model(feat)
        return embedding

    def speaker_diarization_pipline(self, 
                                    audio_path, 
                                    wav_fs=None, 
                                    include_overlap=True,
                                    speaker_num=None):
        
        output_field_labels = self.speaker_diarization(audio_path,wav_fs,include_overlap,speaker_num)
        audio_id = os.path.basename(audio_path).rsplit('.', 1)[0]
        out_json = self.speaker_diarization.diarization_output_json(audio_id, output_field_labels)

        return out_json
    

    def speaker_verification_pipline(self,
                                     audio_path,
                                     sd_result_json,
                                     target_speaker_file_path,
                                     verify_threshold=0.6):
        
        audio_data = load_audio(audio_path,ori_fs=self.fs)
        target_speaker_audio_data = load_audio(target_speaker_file_path,ori_fs=self.fs)
        target_embedding_feature = self.compute_embedding(target_speaker_audio_data)

        data = sd_result_json
        is_target_speaker_idx = []
        for key, value in data.items():
            seg_st, seg_ed, cluster_id = value['start'], value['stop'], value['speaker']
            seg_audio_data = audio_data[:,int(seg_st*self.fs):int(seg_ed*self.fs)]
            if seg_audio_data.shape[1] < self.fs :
                continue
            seg_embedding_feature = self.compute_embedding(seg_audio_data)
            scores = F.cosine_similarity(target_embedding_feature.unsqueeze(0).to(self.device), 
                                         seg_embedding_feature, dim= -1,eps= 1e-12).item()
            if scores > verify_threshold:
                is_target_speaker_idx.append(cluster_id)

        if len(is_target_speaker_idx) == 0:
            return None
        # 计算出现次数最多的值
        unique_values, counts = np.unique(np.array(is_target_speaker_idx), return_counts=True)
        target_speaker_id = unique_values[np.argmax(counts)]
        return target_speaker_id

    def extract_asr_results(self,asr_results_josn,speaker_diarization_file,target_speaker_id):
        results = extract_asr_with_speakers(asr_results_josn, 
                                            speaker_diarization_file, 
                                            target_speaker_id,
                                            )

        return results
 


if __name__ == '__main__':

    os.environ['TORCHAUDIO_USE_FFMPEG'] = '1'
    os.environ['MODELSCOPE_LOG_LEVEL'] = '40'

    GPU_id = 1
    pretrained_model_save_dir = './ckpt'
    target_speaker_audio_file_dir = './file_save_dir/target_speaker_audio'
    audio_input_file_dir = './file_save_dir/meeting_audio'
    asr_results_josn_dir = './file_save_dir/asr_result'

    speaker_diarization_service = SpeakerDiarizationService(
        pretrained_model_save_dir=pretrained_model_save_dir,
        GPU_id=GPU_id   
    )
    # 说话人分离
    # start_time = time.time()
    # 分割模型 15 分钟数据 3.5秒处理时间
    audio_path = os.path.join(audio_input_file_dir,'20240915.wav')
    output_field_labels,sd_result_file = speaker_diarization_service.speaker_diarization_pipline(audio_path,
                                                                                                 include_overlap=True,
                                                                                                 speaker_num=None # 可以选择说话人数量，如果指定，则按照指定数量进行说话人分离
                                                                                                 )
    # print(f'说话人分离时间：{time.time() - start_time}')                                 
    print(f"说话人分割结果:",sd_result_file) 
    target_speaker_file_path = os.path.join(target_speaker_audio_file_dir,'20240915_speaker_3.wav')
    target_speaker_id = speaker_diarization_service.speaker_verification_pipline(audio_path,
                                                                                 sd_result_file,
                                                                                 target_speaker_file_path,
                                                                                 verify_threshold=0.6)
    if target_speaker_id is not None:
        print(f'target_speaker_id: {target_speaker_id}')

        asr_results_josn = os.path.join(asr_results_josn_dir,'asr_20240915.json')
        mapping_result = speaker_diarization_service.extract_asr_results(asr_results_josn,
                                                                      sd_result_file,
                                                                      target_speaker_id,
                                                                      output_json_path=os.path.join(asr_results_josn_dir,'asr_speaker_mapping_20240915.json'),
                                                                      print_to_console=True
                                                                      )
    else:
        print('未检测到目标说话人')