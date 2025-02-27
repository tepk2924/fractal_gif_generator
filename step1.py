import numpy as np
import os
import yaml

#Config file parameters
with open(os.path.join(os.path.dirname(__file__), "step1_option.yml"), "r") as f:
    configs = yaml.safe_load(f)

Xleft = configs['Xleft']
Xright = configs['Xright']
Ydown = configs['Ydown']
Yup = configs['Yup']

Img_height = configs['Img_height']
Img_width = configs['Img_width']
Iternum = configs['Iternum']

HURDLE = 2

#Calculating iteration of fractal
Xcoors = np.tile(np.expand_dims(np.linspace(Xleft, Xright, Img_width), axis=0), (Img_height, 1))
Ycoors = np.tile(np.expand_dims(np.linspace(Yup, Ydown, Img_height), axis=1), (1, Img_width))
complex_seed = Xcoors + Ycoors*1j

complex_calc = np.zeros((Img_height, Img_width), dtype=np.complex128)
calc_result = np.zeros((Img_height, Img_width), np.uint32)

for idx in range(Iternum):
    mask = np.where(np.abs(complex_calc) < HURDLE)
    calc_result[mask] += 1
    complex_calc[mask] = complex_calc[mask]**2 + complex_seed[mask]
    print(f"Iteration {idx + 1}/{Iternum} complete")

#Saving
np.save(os.path.join(os.path.dirname(__file__), "fractal_calc.npy"), calc_result)