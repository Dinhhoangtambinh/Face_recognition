import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

folder_path = 'D:/Workspace_CongViec/FaceAI/Database/Images'
hog_features_file = 'D:/Workspace_CongViec/FaceAI/Database/features.txt'

features_save = 'D:/Workspace_CongViec/FaceAI/Database/features.txt'

MODEL = YOLO('D:/Workspace_CongViec/FaceAI/best.pt')

def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    # Resize the image to reduce computational load
    image_resized = cv2.resize(image, (128, 128))
    hog_features = hog.compute(image_resized)
    print(hog_features)
    print(sum(hog_features != 0))
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
    return names, images

def save_feature_to_file(folder_path, features_save):
    dt_names, dt_images = load_images(folder_path) #Lấy ảnh + lưu tên
    #dt_features = []
    for img, name in zip(dt_images, dt_names):
            dt_feature = extract_hog_features(img)
            dt_feature = adjust_hog_features_size(dt_feature, 10)

            with open(features_save, 'a') as f:
                f.write(f"{name},{dt_feature},\n")

def take_features(features_save):
    with open(features_save) as f:
        lines = f.readlines()
    for line in lines:
        print(line.split(','))
        file_name, file_feature, _ = line.split(',')
        feature_str = file_feature.strip().replace(' ', ',').replace('[', '').replace(']', '')
        feature_list = feature_str.split(',')
        feature_values = [float(value) for value in feature_list if value != '']
        feature_ndarray = np.array(feature_values)
        print(file_name)
        print(feature_ndarray)
print("Nhap chuc nang can thuc hien: ")
while True:
    a = int(input())
    if a == 1:
        save_feature_to_file(folder_path, features_save)
    elif a == 2:
        take_features(features_save)
    elif a == 3:
        MODEL = YOLO('D:/Workspace_CongViec/FaceAI/best.pt')
        results = MODEL.predict('D:/Workspace_CongViec/FaceAI/5.jpg', conf = 0.7)
        # print(results[0].boxes.xywh.tolist()[0])
        for result in results:
            if result.boxes is None:
                continue
            x, y, w, h = result.boxes.xywh.tolist()[0]


            # confidence = result.boxes.conf.tolist()[0]
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            print(x, y, w, h)
            img = cv2.imread('D:/Workspace_CongViec/FaceAI/5.jpg')
            face = img[y:y+h, x:x+w]
            cv2.imshow("Face", face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        break

#Sai trong viec luu file path
