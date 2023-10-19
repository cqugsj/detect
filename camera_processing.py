import base64

import cv2
import mediapipe as mp
import numpy as np
import json
from mqtt_utils import setup_mqtt_client, MQTT_TOPIC
from visualize_utils import visualize
import cProfile
from save_detection_img import save_detected_objects
import os
import time
import requests
import re
import aiohttp
import asyncio
import collections

async def send_post(data, url):
    headers = {'Content-Type': 'application/json'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(data), headers=headers) as response:
                return await response.text()
    except BrokenPipeError:
        print("Broken pipe error occurred when sending POST request.")




# 在 process_camera 函数中
save_directory = "d:/detected_objects"
os.makedirs(save_directory, exist_ok=True)
url = "http://127.0.0.1:7777/setdetections/"


BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def process_camera(camera_id, rtsp_url, show_live=True, save_detection_image=True, send_mqtt=True,
                   send_post_request=True):
    reconnect_attempts_limit = 10
    reconnect_wait_time = 5
    cap = cv2.VideoCapture(rtsp_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 最大缓存帧数，相当于10秒的视频帧
    max_frames = int(20 * fps)  # 假设 fps 是你的视频的帧率

    # 创建一个双端队列作为缓冲区
    buffer = collections.deque(maxlen=max_frames)
    # 创建一个双端队列用于存储帧号
    frame_indices_deque = collections.deque(maxlen=max_frames)

    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}.")
        return
    ip_match = re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', rtsp_url)
    if ip_match:
        ip = ip_match.group()

    def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
        if result.detections != []:
            print('检测结果：摄像头:{},timestamp: {} ms,识别物体: {}'.format(ip, timestamp_ms, result))
            try:
                # 获取当前帧的序号,在哪一帧检测到了数据
                current_frame_index = int(timestamp_ms.value) // 1000
                # 计算开始和结束的帧序号
                # 找到开始和结束帧号在 frame_indices_deque 中的索引
                current_unix_time = int(time.time())

                print("检测帧号" + str(current_frame_index))
                # print(str(frame_indices_deque))
                # print("开始" + str(max(1, current_frame_index - int(5 * fps))))
                # print("结束" + str(min(current_frame_index + int(5 * fps), frame_indices)))
                start_index = frame_indices_deque.index(max(1, current_frame_index - int(5 * fps)))
                end_index = frame_indices_deque.index(min(current_frame_index + int(5 * fps), frame_indices))
            except ValueError as e:
                print("异常信息：" + str(e)+'  检测到帧号：'+str(current_frame_index)+str(frame_indices_deque))
                return

            # 从缓冲区取出这些帧
            relevant_frames = [buffer[i] for i in range(start_index, end_index)]
            # 使用这些帧创建一个新的视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(ip + '_' + str(current_unix_time) + '_' + str(current_frame_index) + '.mp4', fourcc,
                                  fps, (frame_width, frame_height))
            for frame in relevant_frames:
                out.write(frame)
            out.release()

            image_copy = np.copy(output_image.numpy_view())
            if save_detection_image:
                image_path = save_detected_objects(image_copy, ip, result.detections, save_directory)
            serialized_list = []
            for detection in result.detections:
                serialized_detection = {
                    "camera": ip,
                    "bounding_box": vars(detection.bounding_box),
                    "categories": [vars(category) for category in detection.categories],
                    "keypoints": detection.keypoints
                }
                serialized_list.append(serialized_detection)
            _, origine_img = cv2.imencode('.jpg', image_copy)
            origine_img = base64.b64encode(origine_img).decode('utf-8')
            # Get current Unix time
            current_unix_time = int(time.time())
            json_payload_dict = {
                "detections": serialized_list,
                "original_image": origine_img,  # Add finish attribute here
                "detect_time":current_unix_time,
                "detect_frame":current_frame_index
            }
            json_payload = json.dumps(json_payload_dict)
            if send_mqtt and result.detections and len(result.detections) >= 1:
                mqtt_client.publish(MQTT_TOPIC, json_payload)
            if send_post_request:
                data = {
                    'detections': json.dumps(serialized_list),
                    'camera': ip,
                    'image': origine_img,
                    "detect_time": current_unix_time,
                    "detect_frame": current_frame_index
                }
                headers = {'Content-Type': 'application/json'}  # 设置请求头
                response = requests.post(url, data=json.dumps(data), headers=headers)



    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path='d:/efficientdet_lite2.tflite'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        score_threshold=0.7,
        max_results=5,
        result_callback=print_result)
    with ObjectDetector.create_from_options(options) as detector:
        mqtt_client = setup_mqtt_client()
        frame_indices = 0
        while True:
            _, frame = cap.read()
            if frame is None:
                print(f"视频读取失败，重新读取: {rtsp_url}")
                reconnect_attempts = 0
                while True:
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url)
                    if cap.isOpened():
                        print(f"摄像头重新连接成功: {rtsp_url}")
                        _, frame = cap.read()
                        break
                    reconnect_attempts += 1
                    if reconnect_attempts > reconnect_attempts_limit:
                        print(f"摄像头重新连接失败: {rtsp_url}")
                        return
                    time.sleep(reconnect_wait_time)
            frame_indices += 1
            # 把当前帧号添加到帧号队列
            if frame_indices:
                frame_indices_deque.append(frame_indices)
            else:
                print('error1')
            # 把当前帧添加到缓冲区
            if frame is not None:
                buffer.append(frame)
            else:
                print('error2')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(image = mp_image, timestamp_ms = frame_indices)

            if show_live:
                resized_frame = cv2.resize(frame, (400, 300))
                cv2.imshow(f'Camera {camera_id}', resized_frame)
                if cv2.waitKey(1) ==27:
                    break
        cap.release()
        cv2.destroyAllWindows()




def process_camera1(camera_id, rtsp_url,show_live=True,save_detection_image=True,send_mqtt=True,send_post_request=True):
    cap = cv2.VideoCapture(rtsp_url)
    fps =cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print(f"Error: Could not open video stream for camera {camera_id}.")
        return
    # 提取其中的ip
    # 使用正则表达式提取 IP 地址
    ip_match = re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', rtsp_url)
    # 检查是否找到了 IP 地址
    if ip_match:
        ip = ip_match.group()
    #视频流检测到物体的回调函数
    def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
        if result.detections!=[]:
            print('camera:{},timestamp: {} ms,detection result: {}'.format(ip,timestamp_ms, result))
            image_copy = np.copy(output_image.numpy_view())
            if save_detection_image:
                image_path = save_detected_objects(image_copy, ip, result.detections, save_directory)
            #准备数据为mqtt和request
            serialized_list = []
            for detection in result.detections:
                serialized_detection = {
                    "camera": ip,
                    "bounding_box": vars(detection.bounding_box),
                    "categories": [vars(category) for category in detection.categories],
                    "keypoints": detection.keypoints
                }
                serialized_list.append(serialized_detection)
            _, buffer = cv2.imencode('.jpg', image_copy)
            origine_img = base64.b64encode(buffer).decode('utf-8')
            json_payload_dict = {
                "detections": serialized_list,
                "original_image": origine_img  # Add finish attribute here
            }
            json_payload = json.dumps(json_payload_dict)
            if send_mqtt and result.detections and len(result.detections) >= 1:
                mqtt_client.publish(MQTT_TOPIC, json_payload)
            if send_post_request:
                data = {
                    'detections': json.dumps(serialized_list),
                    'camera': ip,
                    'image': origine_img,
                }
                headers = {'Content-Type': 'application/json'}  # 设置请求头
                response = requests.post(url, data=json.dumps(data), headers=headers)

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path='d:/efficientdet_lite2.tflite'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        # category_allowlist=['car'],
        score_threshold=0.5,
        max_results=5,
        result_callback=print_result)

    with ObjectDetector.create_from_options(options) as detector:
        mqtt_client = setup_mqtt_client()
        frame_indices = 0
        while True:
            _, frame = cap.read()
            if frame is None:
                print(f"视频读取失败，重新读取: {rtsp_url}")
                reconnect_attempts = 0
                while True:
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url)
                    # 释放并重新连接摄像头
                    if cap.isOpened():
                        print(f"摄像头重新连接成功: {rtsp_url}")
                        _, frame = cap.read()
                        break
                    else:
                        reconnect_attempts += 1
                        print(f"摄像头重新连接失败，尝试次数: {reconnect_attempts}")
                        # 添加适当的等待时间，避免过于频繁的重试
                        time.sleep(1000)  # 需要导入 time 模块

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_indices += 1
            try:
                detector.detect_async(mp_image, frame_indices)
                if show_live:
                    resized_frame = cv2.resize(frame, (400, 300))
                    cv2.imshow(rtsp_url,resized_frame)
                    if cv2.waitKey(1)==27:
                        break
            except Exception as e:
                print(f"Camera {rtsp_url} encountered an exception: {e}")
                break

    cap.release()
    cv2.destroyAllWindows()
