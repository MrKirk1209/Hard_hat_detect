import random
import torch
from torchvision import transforms
import cv2 as cv
from PIL import Image
from ultralytics import YOLO



# Загрузка обученной модели
# model = YOLO('best.pt')
model = YOLO('hh.pt')
model.fuse()

# source = "https://youtu.be/LNwODJXcvt4"



images='young-man-woman-builder-s-helmets-isolated-white-men-women-background-52549063.webp'
# Путь к вашему тестовому изображению
# test_image_path = 'How_Do_Construction_Workers_Push_Their_Bodies_To_Finish_Olympic_Stadiums_On_Time.mp4'

test_image_path = f'image/{images}'
# Обнаружение объектов на тестовом изображении
# detections = detect_objects(test_image_path)

# # Визуализация результатов
# image = cv2.imread(test_image_path)

results = model.predict(source=test_image_path, conf=0.35)

# prediction_objects = list(results._images_prediction_lst)[0]

# int_labels = prediction_objects.prediction.labels.astype(int)
# class_names = prediction_objects.class_names
# pred_classes = [class_names[i] for i in int_labels]
# print("Num", pred_classes)


# plot = results[0].plot() 
# print(results)
# if(results[0].boxes.id!=None):
#     boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
#     ids = results[0].boxes.id.cpu().numpy.astype(int)
#     for box, id in zip(boxes):
#         random.seed(int(id))
#         color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

#         cv.rectangle(test_image_path, (box[0], box[1]), (box[2], box[3]), color, 2)
#         cv.putText(test_image_path, f'id: {id}', (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#     cv.imshow('frame', test_image_path)

#     cv.waitKey(0)
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
 
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs 
    # print(keypoints) # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
