import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch 
import tensorflow as tf
from ultralytics import YOLO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Đường dẫn 
folder_path = 'D:/Workspace_CongViec/FaceAI/Database/Images'
hog_features_file = 'D:/Workspace_CongViec/FaceAI/Database/features.txt'
features_save = 'D:/Workspace_CongViec/FaceAI/Database/features.txt'


# # Kiểm tra CUDA có sẵn không
# if torch.cuda.is_available():
#     print("CUDA is available. Here are the GPU details:")
#     # Hiển thị số lượng GPU
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     # Liệt kê tên của mỗi GPU
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("CUDA is not available.")

#print("Available devices:", tf.config.list_physical_devices())


# Các hàm liên quan đến đặc trưng
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    # Resize the image to reduce computational load
    image_resized = cv2.resize(image, (128, 128))
    hog_features = hog.compute(image_resized)
    return hog_features.flatten()

def adjust_hog_features_size(features, target_length):
    current_length = len(features)
    if current_length < target_length:
        # If the current length is shorter than the target length, pad zeros to the end
        features = np.pad(features, (0, target_length - current_length), 'constant')
    elif current_length > target_length:
        # If the current length is longer than the target length, truncate the vector
        features = features[-target_length:]
    return features

def load_images(folder_path):
    images = []
    names=[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                # Extract the name before the underscore
                name = filename.split('_')[0]
                names.append(name)
    return images,names

# Camera
def capture_video(MODEL, folder_image, dt_names, dt_features):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
    # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv9 inference on the frame
            results = MODEL.predict(frame, conf = 0.6, save_crop = True)

            for result in results:
                if len(result.boxes) == 0:
                    print("Khong phat hien duoc khuon mat nao!")
                    if cv2.getWindowProperty("Ket Qua Nhan Dien", cv2.WND_PROP_VISIBLE) > 0:
                        cv2.destroyWindow("Ket Qua Nhan Dien")
                    continue
                

                #Thay đổi thông số để quyết định có nhận dạng hay không!
                Recognition = False
                if Recognition:
                    x, y, w, h = result.boxes.xywh.tolist()[0]
                    # confidence = result.boxes.conf.tolist()[0]
                    x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                    print(x, y, w, h)
                    face = frame[y:y+h, x:x+w]

                    face_fea = extract_hog_features(face)
                    face_fea = adjust_hog_features_size(face_fea, 10)

                    # Tính độ tương đồng giữa ảnh chọn và ảnh trong database
                    similarities = []
                    paths = []
                    for dt_feature, dt_name in zip(dt_features, dt_names):
                        similarity = cosine_similarity([face_fea], [dt_feature])
                        similarities.append(similarity[0][0])
                        temp = os.path.join(folder_image, dt_name)
                        paths.append(temp)
                    print(similarities)
                    top_simi = np.argsort(similarities)[::-1][:1]
                    print(dt_names[top_simi[0]])
                    cv2.imshow("Ket Qua Nhan Dien", cv2.imread(paths[top_simi[0]]))

            # Visualize the results on the frame
            annotated_frame = results[0].plot(labels = True, probs = False, conf = True)
            # Display the annotated frame
            cv2.imshow("YOLOv9 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def take_features(features_save):
    with open(features_save) as f:
        lines = f.readlines()
    name = []
    fea = []
    for line in lines:
        file_name, file_feature, _ = line.split(',')
        feature_str = file_feature.strip().replace(' ', ',').replace('[', '').replace(']', '')
        feature_list = feature_str.split(',')
        feature_values = [float(value) for value in feature_list if value != '']
        feature_ndarray = np.array(feature_values)
        name.append(file_name)
        fea.append(feature_ndarray)
    return name, fea

# Main
dt_names, dt_features = take_features(features_save)

MODEL = YOLO('D:/Workspace_CongViec/FaceAI/best.pt')
with torch.no_grad():
    capture_video(MODEL, folder_path, dt_names, dt_features)






            #results[0].save_crop('D:/Workspace_CongViec/FaceAI/Deteced')
            #folder_path = 'D:/Workspace_CongViec/FaceAI/Deteced/Face Detection - v1 V1'

            # Lấy ảnh mới nhất từ danh sách
            #files = os.listdir(folder_path)
            #files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)), reverse=True)
            #need = 0 
            #if files[0] and need == 1:
                #img_name = files[0]
                #img_path = os.path.join(folder_path, img_name)
                #query_face = cv2.imread(img_path)