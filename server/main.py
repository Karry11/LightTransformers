import multiprocessing as mp
from multiprocessing import Process
from http_server import start_http_server, router_worker

# ─── 主入口：启动两个进程 ────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p_http = Process(target=start_http_server)
    p_router = Process(target=router_worker)

    p_http.start()
    p_router.start()

    try:
        p_http.join()
        p_router.join()
    except KeyboardInterrupt:
        p_http.terminate()
        p_router.terminate()
        p_http.join()
        p_router.join()
