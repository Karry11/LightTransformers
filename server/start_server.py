from multiprocessing import Process
from http_server import HttpServer
from server_config import Config

if __name__ == "__main__":
    h = HttpServer(Config.PORT)
    #启动http服务，接收来自user的prompts
    p = Process(target=h.start_http_server(port=Config.PORT))
    p.start()
    p.join()