from torchvision import transforms
import cv2 as cv
from PIL import Image
from ultralytics import YOLO
import os



model = YOLO('hh.pt')
model.fuse()



image_dir = 'image/'
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
counter=1

for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif','webp')):
        image_path = os.path.join(image_dir, image_name)
   
    results = model.predict(source=image_path, conf=0.35)

    
    for result in results:
                
                boxes = result.boxes 
                masks = result.masks  
                keypoints = result.keypoints  
                probs = result.probs  
                obb = result.obb  
                base_name = os.path.splitext(image_name)[0]
                result_image_path = os.path.join(results_dir, f'result_{counter}.jpg')
                result.save(filename=result_image_path)
                counter+=1