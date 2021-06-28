from pathlib import Path
import glob
import shutil
import random

save_dir_path = "data\\conglom_data\\"
retr_dir_path = ".\\data\\cleaning_4_set\\"


# right_droop_files = glob.glob(".\\training_data_min\\*right_droop*\\*.jpg", recursive=True)
# left_droop_files  = glob.glob(".\\training_data_min\\*left_droop*\\*.jpg", recursive=True)
# negative_files    = glob.glob(".\\training_data_min\\*negative*\\*.jpg", recursive=True)
right_droop_dirs = glob.glob(retr_dir_path + "*right_droop*", recursive=True)
left_droop_dirs  = glob.glob(retr_dir_path + "*left_droop*", recursive=True)
negative_dirs    = glob.glob(retr_dir_path + "*negative*", recursive=True)


try:
    shutil.rmtree(save_dir_path)
except:
    pass

Path(save_dir_path + "training_data\\right_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "training_data\\left_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "training_data\\negative\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "validation_data\\right_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "validation_data\\left_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "validation_data\\negative\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "test_data\\right_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "test_data\\left_droop\\").mkdir(parents=True, exist_ok=True)
Path(save_dir_path + "test_data\\negative\\").mkdir(parents=True, exist_ok=True)

random.seed()

test_id = 1

if len(right_droop_dirs) != len(left_droop_dirs):
    print("inequal")
    exit(-1)
if len(right_droop_dirs) != len(negative_dirs):
    print()
    exit(-1)

for i in range(0, len(right_droop_dirs)):

    source_right = right_droop_dirs[i]
    source_left = left_droop_dirs[i]
    source_negative = negative_dirs[i]

    right_files = glob.glob(source_right + "\\*", recursive=True)
    left_files = glob.glob(source_left + "\\*", recursive=True)
    negative_files = glob.glob(source_negative + "\\*", recursive=True)

    x = random.randint(1, 100)

    if x <= 70:
        for file in right_files:
            shutil.copy(file, save_dir_path + "training_data\\right_droop\\")
        for file in left_files:
            shutil.copy(file, save_dir_path + "training_data\\left_droop\\")
        for file in negative_files:
            shutil.copy(file, save_dir_path + "training_data\\negative\\")
    elif x <= 85:
        for file in right_files:
            shutil.copy(file, save_dir_path + "validation_data\\right_droop\\")
        for file in left_files:
            shutil.copy(file, save_dir_path + "validation_data\\left_droop\\")
        for file in negative_files:
            shutil.copy(file, save_dir_path + "validation_data\\negative\\")
    elif x <= 100:
        Path(save_dir_path + "test_data\\right_droop\\" + str(test_id) + "\\test\\").mkdir(parents=True, exist_ok=True)
        Path(save_dir_path + "test_data\\left_droop\\" + str(test_id) + "\\test\\").mkdir(parents=True, exist_ok=True)
        Path(save_dir_path + "test_data\\negative\\" + str(test_id) + "\\test\\").mkdir(parents=True, exist_ok=True)
        for file in right_files:
            shutil.copy(file, save_dir_path + "test_data\\right_droop\\" + str(test_id) + "\\test\\")
        for file in left_files:
            shutil.copy(file, save_dir_path + "test_data\\left_droop\\" + str(test_id) + "\\test\\")
        for file in negative_files:
            shutil.copy(file, save_dir_path + "test_data\\negative\\" + str(test_id) + "\\test\\")
        test_id += 1
