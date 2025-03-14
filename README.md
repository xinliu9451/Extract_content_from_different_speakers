1. 创建model目录，分别下载下列三个模型
    
speech_campplus_sv_zh_en_16k-common_advanced（地址：[CAM++说话人确认-通用-16k-中英文版](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced) ）

speech_fsmn_vad_zh-cn-16k-common-pytorch（地址：[FSMN语音端点检测-中文-通用-16k](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) ）

pyannote_segmentation_3-0.bin（地址：[pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin) ）

2. 在config.py中设置参数

3. bash run_app.sh，开启服务

4. python restful.py 发起测试请求