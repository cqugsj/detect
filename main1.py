import multiprocessing
from camera_processing import process_camera



if __name__ == "__main__":
    rtsp_urls = [
        # "rtsp://admin:hik123456+@117.172.179.191:554/Streaming/Channels/103",
        # "rtsp://admin:2021Sntower++@118.122.182.23:554/Streaming/Channels/103",
        "rtsp://admin:2021Sntower++@119.6.246.30:554/Streaming/Channels/103",
        # "rtsp://admin:2021Sntower++@117.172.179.189:554/Streaming/Channels/103",
        # Add URLs for cameras 5 to 10
    ]
    # 创建一个空列表来存储进程对象
    processes = []
    # 遍历枚举后的rtsp_urls列表
    for i, rtsp_url in enumerate(rtsp_urls):
        # 为每个摄像头创建一个多进程Process对象，目标函数为'process_camera'，参数为(i, rtsp_url)
        process = multiprocessing.Process(target=process_camera, args=(i, rtsp_url,True , True, True))
        # 启动多进程
        process.start()
        # 将Process对象添加到进程列表中
        processes.append(process)
    # 遍历进程列表
    for process in processes:
        # 等待每个进程完成（加入），确保主程序等待所有摄像头完成处理
        process.join()

