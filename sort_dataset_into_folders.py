import os
import pandas as pd
import re
import shutil

dataset_path = os.path.join(os.path.expanduser("~"), "share-all", "derma-data", "archive")
images_path = os.path.join(dataset_path,"ISIC_2019_Training_Input", "ISIC_2019_Training_Input")
print(os.path.isdir(dataset_path))
print(dataset_path)

class_directories = {}
image_name_pattern = re.compile("ISIC_\d{7}.jpg")

image_to_class = pd.read_csv(os.path.join(dataset_path, "ISIC_2019_Training_GroundTruth.csv"), index_col=0)
classes = image_to_class.columns
print(image_to_class[:20])

for img_class in classes:
    class_directory = os.path.join(dataset_path, img_class)
    class_directories[img_class] = class_directory
    print(class_directory)
    print(os.path.isdir(dataset_path))
    if not os.path.isdir(class_directory):
        os.mkdir(class_directory)

print("Classdistribution in the dataset:")
print(image_to_class.mean())

# Clean all filenames from the downsampled suffix
for file in os.listdir(os.fsencode(images_path)):
    image = os.fsdecode(file)
    if not image.endswith(".jpg"):
        continue
    if not image_name_pattern.match(image):
        source_path = os.path.join(images_path, image)
        target_path = os.path.join(images_path, image.replace("_downsampled", ""))
        os.rename(source_path, target_path)

# Move all images to the folders corresponding with its ground truth label
for image, img_class in image_to_class.idxmax(axis=1).items():
    source_path = os.path.join(images_path, image.replace("_downsampled", "") + ".jpg")
    target_path = os.path.join(class_directories[img_class], image.replace("_downsampled", "") + ".jpg")
    if os.path.exists(source_path):
        os.replace(source_path, target_path)
 
os.remove(os.path.join(dataset_path, "ISIC_2019_Training_GroundTruth.csv"))
shutil.rmtree(os.path.join(dataset_path, "ISIC_2019_Training_Input"))


