from pathlib import Path
import glob
import shutil

save_dir_path = "training_data_conglom\\"
b_save_dir_path = "training_data_binary\\"
retr_dir_path = "training_data\\"

right_droop_files = glob.glob(".\\training_data_min\\*right_droop*\\*.jpg", recursive=True)
left_droop_files  = glob.glob(".\\training_data_min\\*left_droop*\\*.jpg", recursive=True)
negative_files    = glob.glob(".\\training_data_min\\*negative*\\*.jpg", recursive=True)
# right_droop_files = glob.glob(".\\training_data\\*right_droop*\\*.jpg", recursive=True)
# left_droop_files  = glob.glob(".\\training_data\\*left_droop*\\*.jpg", recursive=True)
# negative_files    = glob.glob(".\\training_data\\*negative*\\*.jpg", recursive=True)

try:
    shutil.rmtree(save_dir_path)
    shutil.rmtree(b_save_dir_path)
except:
    pass

Path(save_dir_path + "right_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "left_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "negative\\").mkdir(parents=True, exist_ok=True)
Path(b_save_dir_path + "negative\\").mkdir(parents=True, exist_ok=True)
Path(b_save_dir_path + "droop\\").mkdir(parents=True, exist_ok=True)


for source in right_droop_files:
    shutil.copy(source, save_dir_path + "right_droop\\")
    shutil.copy(source, b_save_dir_path + "droop\\")

for source in left_droop_files:
    shutil.copy(source, save_dir_path + "left_droop\\")
    shutil.copy(source, b_save_dir_path + "droop\\")

for source in right_droop_files:
    shutil.copy(source, save_dir_path + "negative\\")
    shutil.copy(source, b_save_dir_path + "negative\\")
