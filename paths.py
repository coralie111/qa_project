import os

plots_dir = 'plots/'
joblib_dir = 'joblib/'

for directory in [joblib_dir]:
    try:
        os.mkdir(directory)
        print("Directory " , directory ,  " Created ") 
    except FileExistsError:
        pass