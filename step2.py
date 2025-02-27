from PIL import Image
import cv2
import numpy as np
import os
import yaml
import imageio

#Intensity value loading
with open(os.path.join(os.path.dirname(__file__), "fractal_calc.npy"), "rb") as f:
    intensity_raw = np.load(f)

#Config file parameters
with open(os.path.join(os.path.dirname(__file__), "step2_option.yml"), "r") as f:
    configs = yaml.safe_load(f)

folderpath = os.path.dirname(__file__)
HSVcolorsetting = configs['HSVColorSetting']
HueSetting = HSVcolorsetting['Hue']
SaturationSetting = HSVcolorsetting['Saturation']
ValueSetting = HSVcolorsetting['Value']
Frames = configs['Frames']
digits = len(str(Frames))
maxval = np.max(intensity_raw)
image_names = [f"$Frame{idx + 1:0{digits}d}.png" for idx in range(Frames)]

Img_height = intensity_raw.shape[0]
Img_width = intensity_raw.shape[1]

#Calculating Intensity Level
intensity = np.where(intensity_raw == maxval, 0, (4*intensity_raw/maxval)**3)
intensity = np.where(intensity > 1, 1, intensity)
intensity = np.expand_dims(intensity, axis=2)

#Generating Images
x = np.tile(np.expand_dims(np.arange(Img_width), 0), (Img_height, 1)) - Img_width/2
y = (np.tile(np.expand_dims(np.arange(Img_height), 0).T, (1, Img_width)) - Img_height/2)[::-1, :]
for idx, (angleframeshift, image_name) in enumerate(zip(np.linspace(0, 2*np.pi, Frames), image_names)):
    angle = np.arctan2(y, x) + angleframeshift
    H = np.expand_dims((HueSetting['Average'] + HueSetting['Deviation']*np.sin(HueSetting['Frequency']*angle + HueSetting['Shift'])).astype(np.uint8), 2)
    S = np.expand_dims((SaturationSetting['Average'] + SaturationSetting['Deviation']*np.sin(SaturationSetting['Frequency']*angle + SaturationSetting['Shift'])).astype(np.uint8), 2)
    V = np.expand_dims((ValueSetting['Average'] + ValueSetting['Deviation']*np.sin(ValueSetting['Frequency']*angle + ValueSetting['Shift'])).astype(np.uint8), 2)
    img_np = (cv2.cvtColor(np.concatenate((H, S, V), axis=2), cv2.COLOR_HSV2RGB)*intensity).astype(np.uint8)
    Image.fromarray(img_np, "RGB").save(os.path.join(folderpath, image_name))
    print(f"done frame {idx + 1:0{digits}d}/{Frames}")

images = [imageio.imread(os.path.join(folderpath, image_name)) for image_name in image_names]
imageio.mimsave(os.path.join(folderpath, "fractal.gif"), images, "GIF", duration=1.0)

for image_name in image_names:
    os.remove(os.path.join(folderpath, image_name))