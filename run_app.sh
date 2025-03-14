#!/bin/bash
# run_transcribe_server.sh

# 使用 Python 读取 config.py 并将参数导出为环境变量
eval $(python3 - <<EOF
import config

# 将 config.py 中的配置打印为环境变量的形式
print(f"export WORKERS={config.WORKERS}")
print(f"export APP_BIND_ADDRESS={config.APP_BIND_ADDRESS}")
print(f"export APP_PORT={config.APP_PORT}")
print(f"export TIMEOUT={config.TIMEOUT}")
print(f"export BACKLOG={config.BACKLOG}")
EOF
)

# 启动 FastAPI 应用
#-w:工作进程数，一般设置为服务器 CPU 核心数的 2-4 倍。
#-k:工作模式，这里使用 UvicornWorker，它是 Uvicorn 的 Gunicorn Worker 类。
#--bind:绑定的 IP 和端口。
#--timeout:工作进程的超时时间，单位为秒。
#--backlog:服务器监听队列的最大长度，也就是当所有 Worker 都在忙碌时，传入的请求会进入监听队列。队列满后，新的连接请求会被拒绝。

gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker app.app:app --bind $APP_BIND_ADDRESS:$APP_PORT --timeout $TIMEOUT --backlog $BACKLOG