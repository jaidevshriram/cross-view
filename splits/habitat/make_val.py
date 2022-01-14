import os
import glob

val = open("/scratch/jaidev/HabitatSplit/val.txt", "r")

scenes = []
for file in val.readlines():
    file = file.strip()

    if os.path.isdir(f"/scratch/jaidev/HabitatMatterport2/new_data/{file}/"):
        scenes.append(f"/scratch/jaidev/HabitatMatterport2/new_data/{file}/0/left_rgb/")
    elif os.path.isdir(f"/scratch_2/jaidev/HabitatMatterport/new_data/{file}/"):
        scenes.append(f"/scratch_2/jaidev/HabitatMatterport/new_data/{file}/0/left_rgb/")
    else:
        print(f"{file} Missing!")

f = open("./val_files.txt", "w")
for scene in scenes:
    print(scene)
    files = os.listdir(scene)
    for file in files:
        file = file.split('.')[0]
        f.write(scene + " " + file + "\n")
