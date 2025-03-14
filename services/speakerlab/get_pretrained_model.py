import os
import torch
from pyannote.audio import Inference, Model


from .utils.builder import build
from .utils.config import Config
from .utils.utils import silent_print

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks



def get_voice_activity_detection_model(device: torch.device=None, cache_dir:str = None):
    cache_dir = os.path.join(cache_dir, 'speech_fsmn_vad_zh-cn-16k-common-pytorch')
    with silent_print():
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection, 
            model=cache_dir, 
            device = 'cpu' if device is None else '%s:%s'%(device.type, device.index) if device.index else device.type,
            disable_pbar=True,
            disable_update=True,
            )
    return vad_pipeline

   
def get_segmentation_model_from_ckpt(device: torch.device=None,cache_dir:str = None):
    ckpt_file_path = os.path.join(cache_dir, "pyannote_sgementation_3-0.bin")
    model = Model.from_pretrained(ckpt_file_path)
    model.eval()
    segmentation = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=32,
        device = device,
        )
    return segmentation

def get_speaker_embedding_model(device:torch.device = None, cache_dir:str = None):
    # CAM++ trained on a large-scale Chinese-English corpus
    conf = {
        'model_id': 'speech_campplus_sv_zh_en_16k-common_advanced',
        'revision': 'v1.0.0',
        'model_ckpt': 'campplus_cn_en_common.pt',
        'embedding_model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'feature_extractor': {
            'obj': 'speakerlab.process.processor.FBank',
            'args': {
                'n_mels': 80,
                'sample_rate': 16000,
                'mean_nor': True,
                },
        }
    }

    cache_dir = os.path.join(cache_dir, conf['model_id'])
    pretrained_model_path = os.path.join(cache_dir, conf['model_ckpt'])
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    # load pretrained model
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device is not None:
        embedding_model.to(device)
    return embedding_model,  feature_extractor
