import os
import cv2
import datetime

def save_detected_objects(frame, camera_id, detections, save_directory):
    saved_image_paths = []
    for idx, detection in enumerate(detections):
        x, y, w, h = detection.bounding_box.origin_x, detection.bounding_box.origin_y, detection.bounding_box.width, detection.bounding_box.height
        object_image = frame[int(y):int(y + h), int(x):int(x + w)]

        # 获取当前时间
        current_time = datetime.datetime.now()

        # 格式化时间为字符串
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")

        # 将时间添加到文件名中
        image_filename = f"camera_{camera_id}_object_{idx + 1}_{detection.categories[0].category_name}_{round(detection.categories[0].score, 2)}_{formatted_time}.jpg"
        image_path = os.path.join(save_directory, image_filename)
        # 保存图像
        cv2.imwrite(image_path, object_image)
        saved_image_paths.append(image_path)
    return saved_image_paths