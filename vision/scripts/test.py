from PIL import Image

img1 = Image.open(r"/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/report/heatmap.png")
img2 = Image.open(r"/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/report/overlay_image.png")

img2.paste(img1, (0, 0))
img2.show()