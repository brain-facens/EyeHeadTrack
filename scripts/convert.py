from PIL import Image 

im = Image.open("/home/eduardo/projects/eyeheadtrack/caixa.jpg")
width, height = im.size 
dpi = im.info.get("dpi", (72, 72))
width_cm = width / dpi[0] * 2.54
height_cm = height / dpi[1] * 2.54 

print(width, height)
print(width_cm, height_cm)