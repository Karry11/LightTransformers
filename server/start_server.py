from multiprocessing import Process
from http_server import HttpServer

if __name__ == "__main__":
    h = HttpServer(8000)
    #启动http服务，接收来自user的prompts
    p = Process(target=h.start_http_server(port=8000))
    p.start()
    p.join()