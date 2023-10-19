import cv2
import mediapipe as mp
import numpy as np
import json
import concurrent.futures
import time
from mqtt_utils import setup_mqtt_client, MQTT_TOPIC
from visualize_utils import visualize
import cProfile

options = mp.tasks.vision.ObjectDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='d:/efficientdet_lite2.tflite'),
    max_results=1,
    score_threshold=0.1,
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)

mqtt_client = setup_mqtt_client()

rtsp_urls = [
    "rtsp://admin:hik123456+@117.172.179.191:554/Streaming/Channels/103",
    "rtsp://admin:2021Sntower++@118.122.182.23:554/Streaming/Channels/103",
    "rtsp://admin:2021Sntower++@119.6.246.30:554/Streaming/Channels/103",
    "rtsp://admin:2021Sntower++@117.172.179.189:554/Streaming/Channels/103",
    # Add URLs for cameras 3 to 10
    # ...
]

caps = [cv2.VideoCapture(rtsp_url) for rtsp_url in rtsp_urls]
fps = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]

# Check if all VideoCapture objects are opened successfully
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video stream for camera {i + 1}.")




def process_camera(cap, detector, fps, mqtt_client, camera_id, process_every_n_frames=5):
    frame_indices = 0
    start_time = time.time()
    frame_counter = 0  # 用于计数处理的帧数

    # 添加性能分析
    pr = cProfile.Profile()
    pr.enable()

    while True:
        _, frame = cap.read()

        if frame is None:
            print(f"Thread {camera_id + 1}: Frame is None. Attempting to reconnect.")
            cap.release()  # 释放原有摄像头资源
            cap = cv2.VideoCapture(rtsp_urls[camera_id])  # 重新连接摄像头
            if not cap.isOpened():
                print(f"Thread {camera_id + 1}: Error: Could not reopen video stream. Exiting thread.")
                break
            else:
                print(f"Thread {camera_id + 1}: Reconnected successfully.")
                continue

        # 处理并显示图像的帧数计数
        frame_counter += 1
        if frame_counter == process_every_n_frames:
            frame_counter = 0  # 重置计数器

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        frame_indices += 1
        frame_timestamp_ms = 1000 * frame_indices / fps

        print(f"Thread {camera_id + 1}: Processing frame {frame_indices} at {frame_timestamp_ms}ms")

        try:
            # 添加性能分析
            pr.clear()
            pr.enable()

            detection_result = detector.detect_for_video(mp_image, int(frame_timestamp_ms))

            if detection_result.detections and len(detection_result.detections) >= 1:
                serialized_list = []
                for detection in detection_result.detections:
                    serialized_detection = {
                        "camera": str(camera_id + 1),
                        "bounding_box": vars(detection.bounding_box),
                        "categories": [vars(category) for category in detection.categories],
                        "keypoints": detection.keypoints
                    }
                    serialized_list.append(serialized_detection)
                json_payload = json.dumps(serialized_list)
                mqtt_client.publish(MQTT_TOPIC, json_payload)

                image_copy = np.copy(mp_image.numpy_view())
                annotated_image = visualize(image_copy, detection_result)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                resized_image = cv2.resize(annotated_image, (400, 300))
                cv2.imshow(f"Camera {camera_id + 1}", resized_image)

                # 重要：等待按键输入，响应键盘事件
                key = cv2.waitKey(1)
                if key == 27:  # 按下 Esc 键退出循环
                    break

            # 添加帧率限制
            processing_time = time.time() - start_time
            if processing_time < 1.0 / fps:
                time.sleep(1.0 / fps - processing_time)

        except Exception as e:
            print(f"Thread {camera_id + 1} encountered an exception: {e}")
            break

        # 添加性能分析
        pr.disable()
        elapsed_time = time.time() - start_time
        print(f"Thread {camera_id + 1}: Elapsed time: {elapsed_time:.2f}s")
        pr.print_stats(sort='cumulative')

    return cap

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        detectors = [mp.tasks.vision.ObjectDetector.create_from_options(options) for _ in range(len(caps))]

        futures = []
        for i, cap in enumerate(caps):
            futures.append(
                executor.submit(process_camera, cap, detectors[i], fps[i], mqtt_client, i)
            )

        # 等待所有线程完成
        concurrent.futures.wait(futures)

    print('All threads finished')

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()