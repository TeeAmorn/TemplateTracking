import cv2
import os

# video_name = 'video.avi'

# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()

video_list = ["draft2_cc_results_43", "draft2_ncc_results_43", "draft2_ssd_results_43",
              "main_cc_results_43", "main_ncc_results_43", "main_ssd_results_43"]

for video_name in video_list:
    video_dir = os.path.join("results", video_name)
    video_save_path = os.path.join("videos", video_name+".avi")
    video = cv2.VideoWriter(video_save_path, 0, 60, (128,96))
    for filename in sorted(os.listdir(video_dir)):
        if filename.endswith(".jpg"):
            video.write(cv2.imread(os.path.join(video_dir, filename)))
    cv2.destroyAllWindows()
    video.release()
