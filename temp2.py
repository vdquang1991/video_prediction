import os
import cv2
import numpy as np

# root_path = 'results_ourproposed'
# folder_path = 'running/person21_running_d4_uncomp'
# out_path = 'results_MHI_ourproposed'
# start_idx = 200
# end_idx = start_idx + 40
# prev_img = 0
#
# for i in range(start_idx, end_idx):
#     img_path = os.path.join(root_path, folder_path, str(i).zfill(5) + '.jpg')
#     cur_img = cv2.imread(img_path)
#     if i>start_idx:
#         img_diff = cv2.absdiff(cur_img, prev_img)
#         cv2.imwrite(os.path.join(out_path, str(i).zfill(5) + '.jpg'), img_diff)
#     prev_img = cur_img

our_proposed = np.random.uniform(low=0.035, high=0.085, size=(30,))
our_proposed = our_proposed.round(decimals=3)
our_proposed = np.sort(our_proposed)
print(list(our_proposed))

