# coding=utf-8
import os
import sys
from loguru import logger

log_directory =  "logs" # 定义日志目录
os.makedirs(log_directory, exist_ok=True)  # 如果目录不存在则创建

# 移除默认的日志配置
logger.remove()

# 日志文件路径
log_file_path = os.path.join(log_directory, "app.log")

# 添加文件日志处理器
logger.add(
    log_file_path,
    rotation="1 week",  # 日志文件每周轮换一次
    retention="4 weeks",  # 只保留最近4周的日志
    level="INFO",  # 日志级别
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    compression="zip",  # 压缩旧的日志文件
)

# 添加控制台日志处理器
logger.add(
    sys.stdout,
    level="INFO",  # 日志级别
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
)