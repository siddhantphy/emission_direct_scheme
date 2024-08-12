import os

MAIN_DIRECTORY = os.path.dirname(os.path.dirname(__file__))


def get_full_path(*path):
    full_path = os.path.join(MAIN_DIRECTORY, *path)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    return full_path


def filename_contains_datestamp(fn):
    if fn[:8].isdigit() and fn[8] == "_" and fn[9:15].isdigit() and fn[15] == "_":
        return True
    else:
        return False


def detect_filenames(folder, fn, export_full_names=False):
    main_file = fn[16:] if filename_contains_datestamp(fn) else fn
    main_file_compare = main_file
    if "_cut_off_time-" in main_file_compare:
        main_file_compare = main_file
        main_file_compare = main_file_compare.split("_cut_off_time-")[0] + "_cut_off_time-" + str(round(float(main_file_compare.split("_cut_off_time-")[1]), 15))
    main_file_present = False
    sub_files = []

    for file in os.listdir(get_full_path(folder)):
        if file.endswith(".csv"):
            file_compare = file
            if "_cut_off_time-" in file:
                file_compare = file.split(".csv")[0]
                if "_failed" in file_compare:
                    file_compare = file_compare.split("_failed")[0]
                    failed_txt = "_failed"
                else:
                    failed_txt = ""
                file_compare = file_compare.split("_cut_off_time-")[0] + "_cut_off_time-" + str(round(float(file_compare.split("_cut_off_time-")[1].split("_merged")[0]), 15)) + failed_txt + ".csv"
            if filename_contains_datestamp(file):
                if file_compare[16:-4] == main_file_compare:
                    sub_files.append(file[:-4]) if export_full_names else sub_files.append(file[:15])
            if file_compare[:-4] == main_file_compare:
                print(f"Found file: {file}")
                if main_file_present is True:
                    raise FileExistsError(f"There is more than one main file {main_file} in the specified folder {folder}.")
                main_file_present = True
    return main_file, main_file_present, sub_files


def detect_filenames_folder(folder):
    files = {}
    for file in os.listdir(get_full_path(folder)):
        if filename_contains_datestamp(file) and file.endswith(".csv"):
            main_file = file[16:-4]
            if main_file not in files:
                files[main_file] = [file[:15]]
            else:
                files[main_file].append(file[:15])
    return files