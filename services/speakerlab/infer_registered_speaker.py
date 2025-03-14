import os
import warnings
import torch
import numpy as np
from datetime import datetime
from loguru import logger
from speakerlab.utils.compute_snr import calculate_snr
from speakerlab.utils.fileio import load_audio

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
warnings.filterwarnings("ignore")
 


class RegisterSpeaker():
    def __init__(self,
                 embedding_model,
                 embedding_feature_extractor,
                 my_embedding_save_dir,
                 min_audio_duration=10,
                 min_snr_threshold=5,
                 device=None):
        self.embedding_model = embedding_model
        self.embedding_feature_extractor = embedding_feature_extractor
        self.my_embedding_dir_path = my_embedding_save_dir
        self.min_audio_duration = min_audio_duration
        self.min_snr_threshold = min_snr_threshold
        self.device = device        

    @staticmethod
    def normalize(audio, target_level=-25):
        # normalize the signal to the target
        rms = (audio ** 2).mean() ** 0.5
        scalar = 10 ** (target_level / 20) / (rms + 1e-10)
        return audio * scalar


    def pre_register(self,target_speaker_audio_root_path):
        audio_data = load_audio(target_speaker_audio_root_path,ori_fs=16000)
        if audio_data.shape[1] < 10 * 16000:
            logger.info(f"音频数据长度小于10秒，请重新录制")
            return None
        audio_data = self.normalize(audio_data, target_level=-25)
        # 计算音频文件的SNR
        # 这里涉及到计算平稳噪声，如何能够准确获取到背景噪声可以优化，最好是通过软件提醒，比如给用户一个指令后再读文字。
        target_speaker_audio_snr = calculate_snr(target_speaker_audio_root_path)
        if target_speaker_audio_snr < self.min_snr_threshold:
            logger.info(f"当前环境比较嘈杂，请在安静的场景下重新注册声纹信息。")
            return None
        return audio_data


    def register_speaker_embedding(self,audio_data):
        """
        注册说话人声纹信息
        """
        # 计算声纹特征
        # compute feat
        feat = self.embedding_feature_extractor(audio_data).unsqueeze(0).to(self.device) # torch
        # compute embedding
        with torch.no_grad():
            current_target_speaker_embedding_feature = self.embedding_model(feat).detach().squeeze(0).cpu().numpy()
        # 保存声纹信息
        
        return current_target_speaker_embedding_feature

    def __call__(self,target_speaker_audio_root_path):
        # 预处理，排除一些环境因素可能导致的声纹信息质量不高
        audio_data = self.pre_register(target_speaker_audio_root_path)
        if audio_data is None:
            return
        
        # 判断是否注册过声纹信息,目标文件夹地址仅保存用户一个人的声纹特征，如果有其他人则保存在另一个
        target_embedding_feature_list = os.listdir(self.my_embedding_dir_path) # 暂时认为最多有一个声纹信息
        assert len(target_embedding_feature_list) < 2, f"用户声纹信息文件夹中存在多个声纹信息文件，暂未支持存在多个声纹信息文件，请查明原因！"

        # #根据音频文件名和注册时间生成声纹信息文件名
        target_speaker_audio_name = os.path.basename(target_speaker_audio_root_path)
        register_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        target_embedding_feature_name = f"MyEmbedding_{register_time}_{target_speaker_audio_name.split('.')[0]}.npy"


        # 判断是否注册过声纹信息,目标文件夹地址仅保存用户一个人的声纹特征，如果有其他人则保存在另一个
        target_embedding_feature_list = os.listdir(self.my_embedding_dir_path) # 暂时认为仅有一个声纹信息

        if len(target_embedding_feature_list) == 0:
            print(f"用户未注册过声纹信息,开始注册声纹信息")
            target_embedding_feature = self.register_speaker_embedding(audio_data)
            
        else:
            target_embedding_feature = target_embedding_feature_list[0]
            is_user_embedding = target_embedding_feature.endswith('.npy') and target_embedding_feature.split('_')[0]=='MyEmbedding'
            if is_user_embedding: # 判断是否为用户本人的声纹特征文件
                print(f"用户已注册过声纹信息,注册声纹文件为名为：{target_embedding_feature_list[0]}")
                # 如果用户选择替换声纹信息，则删除原有声纹信息，并重新注册声纹信息
                if input(f"是否替换注册声纹信息？(y/n): ").lower() == 'y':
                    print(f"开始替换注册声纹信息")
                    target_embedding_feature= self.register_speaker_embedding(audio_data)
                    os.system(f"rm {os.path.join(self.my_embedding_dir_path, target_embedding_feature_list[0])}")
                else:
                    print("保留现有声纹信息，注册取消。")
                    return None
            else:
                print(f"存在未知声纹信息，请确定是否为用户注册用户。")
        
        target_embedding_feature_path = os.path.join(self.my_embedding_dir_path, target_embedding_feature_name)
        np.save(target_embedding_feature_path, target_embedding_feature)
        print(f"声纹信息注册成功，注册声纹文件地址为：{target_embedding_feature_path}")
        return target_embedding_feature_path

