import cv2
import math

def jpeg_compress(image, quality):
    encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    return decoded_image

def resize_cover(image, size):
    # Basic image size
    h_img, w_img = image.shape[:2]
    w_target, h_target = size
    # Calculate ratio to ensure temporary image size >= target size
    ratio = max(w_target / w_img, h_target / h_img)
    w_temp = math.ceil(w_img * ratio)
    h_temp = math.ceil(h_img * ratio)
    
    # Resize
    temp_img = cv2.resize(image, (w_temp, h_temp), interpolation=cv2.INTER_LANCZOS4)
    
    # Calculate start and end coordinates
    start_x = max(0, (w_temp - w_target) // 2)
    start_y = max(0, (h_temp - h_target) // 2)
    end_x = start_x + w_target
    end_y = start_y + h_target
    # If end exceeds the boundary of temp_img, force it to shrink
    if end_x > w_temp:
        start_x -= (end_x - w_temp)
        end_x = w_temp
    if end_y > h_temp:
        start_y -= (end_y - h_temp)
        end_y = h_temp

    return temp_img[start_y:end_y, start_x:end_x]
