import os

os.system("pip install -r requirements.txt")

path_1 = os.sep.join(["data", "images"])
path_2 = os.sep.join(["data", "labels"])
os.system(f"mkdir {path_1}")
os.system(f"mkdir {path_2}")

path = os.sep.join(["UNet", "models"])
os.system(f"mkdir {path}")

path = os.sep.join(["UNet", "visualizations"])
os.system(f"mkdir {path}")