import os
import sys

import cv2

write_to_folder = '/home/venky/CogVLM/test_sets/testset_vids/IMG_1131_frames'

def get_frames():
    cap = cv2.VideoCapture('/home/venky/CogVLM/test_sets/testset_vids/IMG_1131.MOV')
    i = 0
    frame_skip = 30
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            # cv2.imwrite('test_'+str(frame_count*frame_skip)+'.jpg', frame)
            # cv2.imwrite('/home/venky/CogVLM/test_sets/testset_vids/img_1129_frames/1129_frame_'+str(frame_count*frame_skip)+'.jpg', frame)
            os.makedirs(write_to_folder, exist_ok=True)
            cv2.imwrite(os.path.join(write_to_folder, '1131_frame_'+str(frame_count*frame_skip)+'.jpg'), frame)
            i = 0
            continue
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_frames()