from multiprocessing import Process
from http_server import run_http_server
import time


# 主进程
if __name__ == "__main__":
    #启动子进程接收prompt数据
    print("[Main] Launching HTTP server subprocess...")
    p = Process(target=run_http_server)
    p.start()

    # 模拟主进程继续执行其他逻辑
    try:
        while True:
            print("[Main] Main process is running...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("[Main] Terminating...")
        p.terminate()
        p.join()
