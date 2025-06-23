#!/usr/bin/env python3
import subprocess
import redis

def run_sudo(cmd, password):
    result = subprocess.run(
        ["sudo", "-S"] + cmd,
        input=password,
        text=True,
        capture_output=True
    )
    return result.returncode, result.stdout, result.stderr

def ensure_redis_running(password="134125\n"):
    # 1) 检查状态
    code, _, _ = run_sudo(["systemctl", "status", "redis"], password)
    if code == 0:
        print("Redis 已在运行，无需操作。")
    else:
        print("Redis 未运行，尝试启动...")
        code, out, err = run_sudo(["systemctl", "start", "redis"], password)
        if code == 0:
            print("Redis 启动成功。")
        else:
            print(f"启动失败（exit {code}），错误信息：\n{err}")


# ensure_redis_running()
# from common.config import  Config
# config_params = Config
# ─── Redis 连接池 ───────────────────────────────────────────────────────────
# redis_pool = redis.ConnectionPool(
#     host=config_params.REDIS_HOST,
#     port=config_params.PORT,
#     db=0,
#     max_connections=config_params.REDIS_MAX_CONNECTIONS,
# )
