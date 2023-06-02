import cv2
import os
import numpy as np

input_root = 'test'
out_folder = 'results_ourproposed'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

class_folder = os.listdir(input_root)

for c in class_folder:
    video_folder = os.listdir(os.path.join(input_root, c))
    for v in video_folder:
        video_path = os.path.join('predict_video_LMC', c, v)
        if os.path.exists(video_path):
            img_list = sorted(os.listdir(video_path))
            start_index = int(img_list[0].split('.')[0])
            end_index = start_index + 40
            for index in range(start_index, end_index):
                img_name = str(index).zfill(5) + '.jpg'
                img_path = os.path.join(input_root, c, v, img_name)
                img1 = cv2.imread(img_path)
                img_path = os.path.join('predict_video_LMC', c, v, img_name)
                img2 = cv2.imread(img_path)
                img2 = cv2.resize(img2, (120, 120))
                img = np.asarray(img1, dtype=np.float) * 3 + np.asarray(img2, dtype=np.float)
                img /= 4.
                img = np.asarray(img, dtype=np.uint8)
                img[img<80] += 20
                # img1[img1<80] +=20
                if not os.path.exists(os.path.join(out_folder, c)):
                    os.mkdir(os.path.join(out_folder, c))
                if not os.path.exists(os.path.join(out_folder, c, v)):
                    os.mkdir(os.path.join(out_folder, c, v))
                img_path = os.path.join(out_folder, c, v, img_name)
                cv2.imwrite(img_path, img)



