def process_camera(camera_id, rtsp_url, show_live=True, save_detection_image=True, send_mqtt=True,
                   send_post_request=True):
    reconnect_attempts_limit = 10
    reconnect_wait_time = 5
    cap = cv2.VideoCapture(rtsp_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 最大缓存帧数，相当于10秒的视频帧
    max_frames = int(10 * fps)  # 假设 fps 是你的视频的帧率

    # 创建一个双端队列作为缓冲区
    buffer = collections.deque(maxlen=max_frames)

    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}.")
        return
    ip_match = re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', rtsp_url)
    if ip_match:
        ip = ip_match.group()

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path='d:/efficientdet_lite2.tflite'),
        running_mode=VisionRunningMode.LIVE_STREAM,
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
            # 把当前帧添加到缓冲区
            buffer.append(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(image = mp_image, timestamp_ms = frame_indices)
            print(frame_indices)

            if show_live:
                resized_frame = cv2.resize(frame, (400, 300))
                cv2.imshow(f'Camera {camera_id}', resized_frame)
                if cv2.waitKey(1) ==27:
                    break
        cap.release()
        cv2.destroyAllWindows()
