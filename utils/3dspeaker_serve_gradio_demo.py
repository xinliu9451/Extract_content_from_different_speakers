import os
import gradio as gr

import time
from speakerlab.infer_sd_service import SpeakerDiarizationService

# 环境设置
os.environ['TORCHAUDIO_USE_FFMPEG'] = '1'
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'

# 配置参数
GPU_id = 1  # 使用的GPU ID
pretrained_model_save_dir = './ckpt'  # 预训练模型保存目录
temp_dir = './file_save_dir/temp'  # 临时文件目录


# 初始化服务
speaker_diarization_service = SpeakerDiarizationService(
    pretrained_model_save_dir=pretrained_model_save_dir,
    GPU_id=GPU_id
)


def process_audio(target_speaker_audio, meeting_audio, asr_json_file, verify_threshold=0.6, include_overlap=True, speaker_num=0):
    """
    处理音频文件并执行说话人分离和识别
    
    参数:
        target_speaker_audio: 目标说话人音频文件路径
        meeting_audio: 会议音频文件路径
        asr_json_file: ASR结果JSON文件路径
        verify_threshold: 说话人验证阈值
        include_overlap: 是否包含重叠说话
        speaker_num: 说话人数量，0表示自动检测
    
    返回:
        result_text: 处理结果文本
        result_json: 结果JSON字符串
        result_json_path: 结果JSON文件路径
    """
    result_text = "处理日志:\n"
    if speaker_num == 0:
        speaker_num = None
    try:
        # Print debug info
        result_text += f"目标说话人音频: {target_speaker_audio}\n"
        result_text += f"会议音频: {meeting_audio}\n"
        
        # 2. 说话人分离
        start_time = time.time()
        result_text += "正在进行说话人分离...\n"
        sd_result_file_path = os.path.join(temp_dir, f'sd_result_{int(time.time())}.json')
        sd_result_json = speaker_diarization_service.speaker_diarization_pipline(
            audio_path=meeting_audio,
            include_overlap=include_overlap,
            speaker_num=speaker_num,
            save_to_json=True,
            sd_result_file_path=sd_result_file_path
        )
        result_text += f"说话人分离完成，耗时: {time.time() - start_time:.2f}秒\n"
        result_text += f"说话人分割结果文件: {sd_result_file_path}\n\n"
        
        # 3. 说话人验证
        start_time = time.time()
        result_text += "正在进行说话人验证...\n"
        target_speaker_id = speaker_diarization_service.speaker_verification_pipline(
            audio_path=meeting_audio,
            sd_result_json=sd_result_json,
            target_speaker_file_path=target_speaker_audio,
            verify_threshold=verify_threshold
        )
        result_text += f"说话人验证完成，耗时: {time.time() - start_time:.2f}秒\n"
        
        # 4. 提取ASR结果
        if target_speaker_id is not None:
            result_text += f"目标说话人ID: {target_speaker_id}\n"
            start_time = time.time()
            result_text += "正在提取ASR结果...\n"
            
            # 生成输出文件路径
            output_json_path = os.path.join(temp_dir, f'asr_speaker_mapping_{int(time.time())}.json')
            
            # 提取ASR结果
            mapping_result = speaker_diarization_service.extract_asr_results(
                asr_json_file,
                sd_result_file_path,
                target_speaker_id,
                output_json_path=output_json_path,
                print_to_console=False
            )
            
            result_text += f"ASR结果提取完成，耗时: {time.time() - start_time:.2f}秒\n"
            result_text += f"结果文件已保存至: {output_json_path}\n"
            
            # 格式化结果用于显示
            formatted_results = []
            for item in mapping_result:
                speaker = "me" if item.get("speaker") == "me" else f"Speaker {item.get('speaker')}"
                formatted_results.append(f"[{item.get('time')}] {speaker}: {item.get('text')}")
            
            result_text += "\n处理结果预览:\n" + "\n".join(formatted_results[:10])
            if len(formatted_results) > 10:
                result_text += f"\n... 共{len(formatted_results)}条记录"
            
            # 返回结果文件路径
            with open(output_json_path, 'r', encoding='utf-8') as f:
                result_json = f.read()
            
            return result_text, result_json, output_json_path
        
        else:
            result_text += "未检测到目标说话人\n"
            return result_text, None, None
            
    except Exception as e:
        result_text += f"处理过程中出错: {str(e)}\n"
        return result_text, None, None

# 创建Gradio界面
with gr.Blocks(title="3D Speaker Diarization Service") as demo:
    gr.Markdown("# 3D Speaker Diarization Service")
    gr.Markdown("使用此服务进行说话人分离，并提取目标说话人的发言内容")
    
    with gr.Row():
        with gr.Column():
            # 输入组件
            target_speaker_audio = gr.Audio(label="目标说话人音频", type="filepath", sources=["upload", "microphone"])
            meeting_audio = gr.Audio(label="会议音频",type="filepath",sources=["upload", "microphone"])
            asr_json_file = gr.File(label="ASR结果JSON文件",file_types=[".json"])
            
            with gr.Row():
                verify_threshold = gr.Slider(
                    label="说话人验证阈值",
                    minimum=0.1,
                    maximum=0.99,
                    value=0.6,
                    step=0.05
                )
                include_overlap = gr.Checkbox(label="包含重叠说话",value=True)
            
            speaker_num = gr.Number(label="说话人数量（0为自动检测）",value=0)
            
            process_btn = gr.Button("开始处理", variant="primary")
            
        with gr.Column():
            # 输出组件
            result_text = gr.Textbox(label="处理结果",lines=20)
            result_json = gr.JSON(label="JSON结果")
            result_file = gr.File(label="下载结果文件")
    
    # 连接处理函数
    process_btn.click(
        fn=process_audio,
        inputs=[
            target_speaker_audio,
            meeting_audio,
            asr_json_file,
            verify_threshold,
            include_overlap,
            speaker_num
        ],
        outputs=[
            result_text,
            result_json,
            result_file
        ]
    )
    
    gr.Markdown("### 说明")
    gr.Markdown("""
    1. 上传目标说话人的音频文件（用于声纹注册）
    2. 上传需要分析的会议音频文件
    3. 上传会议音频对应的ASR结果JSON文件
    4. 调整参数（可选）
    5. 点击"开始处理"按钮
    6. 处理完成后，可以查看和下载结果
    """)

# 启动Gradio应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)