import os
import warnings
import torch
import numpy as np


from speakerlab.infer_registered_speaker import RegisterSpeaker
from speakerlab.get_pretrained_model import get_speaker_embedding_model

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
warnings.filterwarnings("ignore")



class ARGS:
    def __init__(self):
        self.pretrained_model_save_dir = "./ckpt"
        self.audio_in = "./file_save_dir/meeting_audio/test_12.wav"
        self.sgementation_ckpt= "./ckpt/pyannote_sgementation_3-0.bin" 
        self.my_embedding_save_dir = "./file_save_dir/EmbeddingFeatures/MyEmbedding"
        self.out_dir="./file_save_dir/speaker_segmentation_result"
        self.out_type="json"
        self.include_overlap=True
        self.disable_progress_bar=False
        self.inference_gpu_id = 1
        self.nprocs=1
        self.speaker_num=None


if __name__ == '__main__':
    args = ARGS()
    inference_device = torch.device('cuda:%d'%(args.inference_gpu_id))
    embedding_model, embedding_feature_extractor = get_speaker_embedding_model(inference_device, args.pretrained_model_save_dir)
    my_embedding_save_dir = args.my_embedding_save_dir

    registered_speaker_pipline = RegisterSpeaker(
                 embedding_model,
                 embedding_feature_extractor,
                 my_embedding_save_dir,
                 min_audio_duration=10,
                 min_snr_threshold=5,
                 device=None)
    target_embedding_feature_path = registered_speaker_pipline(args.audio_in)
    print(f"Target embedding feature path: {target_embedding_feature_path}")