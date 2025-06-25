
# This code allows to establish the appropriate project structure, so that the images can be fed into an Ultralytics YOLO model

import os
import pandas as pd
import shutil
import yaml
from utils import copy_files


# Function to move files
def move_files(files_list, source_folder, destination_folder):
    for file_name in files_list:
        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            shutil.move(source_file_path, destination_folder)
            print(f'{file_name} moved to {destination_folder}')
        else:
            print(f'{file_name} is not located in {source_folder}')

#Define function to create the txt files for every image
def create_txt_labels(df, labels_folder, img_width, img_height):
    for index, row in df.iterrows():
        file_name = row['image_name'] 
        labels = row['BoxesString'] 
        # Skip file creation if the label is "no_box"
        if labels == "no_box":
            print(f'Creation of file {file_name} omitted due to "no_box" label.')
            continue

        # Get file name without .png extension 
        base_name = os.path.splitext(file_name)[0]
        
        # Create .txt file for the image labels
        label_file_path = os.path.join(labels_folder, f"{base_name}.txt")
        
        with open(label_file_path, 'w') as f:
            # Separate the labels of each object in the image
            objects = labels.split(';')
            
            for obj in objects:
                x_min, y_min, x_max, y_max = map(int, obj.split())

                # Calculate x_center, y_center, width and height, normalizing between 0 and 1
                x_center = (x_min + x_max) / (2*img_width)
                y_center = (y_min + y_max) / (2*img_height)
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # Write each label in the file and add class 0 at the beginning
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

        print(f'Label file for {file_name} created at {label_file_path}')


# Path of the csv files that contain data for the train, valid and test splits
csv_train_path = "../gwhd_2021/competition_train.csv"
csv_valid_path = "../gwhd_2021/competition_val.csv"
csv_test_path = "../gwhd_2021/competition_test.csv"

# Path for the original folder containing all the gwhd 2021 images
img_source_folder = "../gwhd_2021/images/"

# New folders where the images will be moved to according to the csv files
destination_train = "../gwhd_2021/Original/train/images/"
destination_valid = "../gwhd_2021/Original/valid/images/"
destination_test = "../gwhd_2021/Original/test/images/"

# New folders for the label files in YOLO xyhw format 
labels_train = "../gwhd_2021/Original/train/labels/"
labels_valid = "../gwhd_2021/Original/valid/labels/"
labels_test = "../gwhd_2021/Original/test/labels/"

# Create the above directories in case they don't exist 
os.makedirs(destination_train, exist_ok=True)
os.makedirs(destination_valid, exist_ok=True)
os.makedirs(destination_test, exist_ok=True)

os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_valid, exist_ok=True)
os.makedirs(labels_test, exist_ok=True)

# Read the csv files
train = pd.read_csv(csv_train_path)
valid = pd.read_csv(csv_valid_path)
test = pd.read_csv(csv_test_path)


# Fix an issue with duplicate images in the csv for train and test 
mod1 = (train["image_name"] == "1961bcf453d5b2206c428c1c14fe55d1f26f3c655db0a2b6a83094476e8edb5b.png") & (train["domain"] == "Arvalis_10")
mod2 = (train["image_name"] == "d88963636d49127bda0597ef73f1703e92d6f111caefc44902d5932b8cd3fa94.png") & (train["domain"] == "Arvalis_10")
mod3 = (test["image_name"] == "da9846512ff19b8cd7278c8c973f75d36de8c4eb4e593b8285f6821aae1f4203.png") & (test["domain"] == "CIMMYT_1")
train.loc[mod1, "image_name"] = "b588e1b55fbf8c4c6af08886013c0c36b70dd617fb1a5070829295a5c3ab31a8.png"
train.loc[mod2, "image_name"] = "8cc5870f73527da07937acc806002e1272e6200095c656ca326a680a90fab507.png"
test.loc[mod3, "image_name"] = "094dcc9098204e6f751504515f3b0a7b5f5ad500a8bd2ec10124ed3d4fdbb6ed.png"

# Get the file names for the images
train_files = train["image_name"].tolist() 
valid_files = valid["image_name"].tolist() 
test_files = test["image_name"].tolist() 

# Moving the files to the new folders 
move_files(train_files, img_source_folder, destination_train)
move_files(valid_files, img_source_folder, destination_valid)
move_files(test_files, img_source_folder, destination_test)


#Creating the label files in every corresponding folder
create_txt_labels(train, labels_train, img_width=1024, img_height=1024)
create_txt_labels(valid, labels_valid, img_width=1024, img_height=1024)
create_txt_labels(test, labels_test, img_width=1024, img_height=1024)

####################################################################################################################################################################
# Creation of a YAML file with the dataset config for the YOLO model 
# Define the data that will be written in the yaml file
data = {
    'path': '../gwhd_2021/Original', #root dir to dataset 
    'train': 'train/images', #train images relative to 'path'
    'val': 'valid/images', #val images relative to 'path'
    'test': 'test/images', #test images relative to 'path'
    'names': {
        0: 'wheat_head'
    } 
}

# Path for the yaml file
yaml_file_path = '../gwhd_2021/gwhd_2021.yaml'

# Create the yaml file
with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, default_flow_style=False, sort_keys=False)

print(f'YAML file created at {yaml_file_path}')


# Another YAML file is created for the copy of the dataset where the images are modified by OT domain adaptation
# Define the data that will be written in the yaml file
data = {
    'path': '../gwhd_2021/OT', #root dir to dataset 
    'train': 'train/images', #train images relative to 'path'
    'val': 'valid/images', #val images relative to 'path'
    'test': 'test/images', #test images relative to 'path'
    'names': {
        0: 'wheat_head'
    } 
}

# Path for the yaml file
yaml_file_path = '../gwhd_2021/gwhd_2021_OT.yaml'

# Create the yaml file
with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, default_flow_style=False, sort_keys=False)

print(f'YAML file created at {yaml_file_path}')

##################################################################################################################3
# Copy the labels to the new folders where the OT modified images will be stored

# Path to the source folders
source_train = "../gwhd_2021/Original/train/labels/"
source_valid = "../gwhd_2021/Original/valid/labels/"
source_test = "../gwhd_2021/Original/test/labels/"

# Path to the destination folders
destination_train = "../gwhd_2021/OT/train/labels/"
destination_valid = "../gwhd_2021/OT/valid/labels/"
destination_test = "../gwhd_2021/OT/test/labels/"

# Create destination folders in case they don't exist
os.makedirs(destination_train, exist_ok=True)
os.makedirs(destination_valid, exist_ok=True)
os.makedirs(destination_test, exist_ok=True)

# Get file list for the files in the source folders
files_train = os.listdir(source_train)
files_valid = os.listdir(source_valid)
files_test = os.listdir(source_test)

# Make the copy of the label files and store them in new folders for the OT images
copy_files(source_train, files_train, destination_train)
copy_files(source_valid, files_valid, destination_valid)
copy_files(source_test, files_test, destination_test)
