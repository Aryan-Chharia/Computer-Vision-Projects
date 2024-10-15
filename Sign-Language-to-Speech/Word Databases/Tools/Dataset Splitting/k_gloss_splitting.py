"""
The script allows to divide the WLASL dataset into sub-datasets. The division
is made according to the order indicated in the JSON file. This file is made
available by the authors of WLASL dataset.
Usage: python k_gloss_splitting.py param1 param2
 - param1: path to the full dataset (e.g. ./WLASL_full/)
 - param2: number of glosses to be considered for the split (e.g. 2000)
"""
import json
import os
import shutil
import sys

import cv2
from tqdm import tqdm

# global variables
PATH_JSON = './WLASL_v0.3.json'


def main():
    try:
        # command line inputs
        path_dataset = sys.argv[1]
        glosses = int(sys.argv[2])
        if not 1 <= glosses <= 2000:
            raise ValueError('\nInsert an integer: 1~2000')

        # set the name of dir that will contain the spilt
        path_k_glosses_dir = './WLASL_' + str(glosses) + '/'

        print('[log] > START DATASET PROCESSING ...\n')
        dataset_processing(glosses, path_k_glosses_dir, path_dataset)
        show_info(path_k_glosses_dir)
        print('\n[log] > DONE!')

    except ValueError:
        print('Insert an integer: 1~2000')


def dataset_processing(glosses, path_k_glosses_dir, path_dataset):
    # read the json as a list of dictionaries
    wlasl_json = read_json(PATH_JSON)

    # split the videos in train, val and test
    splitted_videos = splitting_train_val_test(wlasl_json, glosses)

    # create dirs in which we'll store the videos
    make_target_dirs(wlasl_json, glosses, path_k_glosses_dir)

    # copy the videos in their own dir
    save_in_dirs(path_dataset, path_k_glosses_dir, splitted_videos)


def read_json(file_path):
    with open(file_path) as f:
        wlasl_json = json.load(f)
    return wlasl_json


def splitting_train_val_test(json_file, glosses):
    print('[log] > Splitting videos in train, val and test ...')
    # save in a dictionary the 'video_id' - ['target_dit', ] pair
    videos_dict = {}
    for k, gloss in tqdm(enumerate(json_file)):  # iterate through each gloss
        if k < glosses:
            videos = gloss['instances']  # get all videos as array
            for video in videos:
                video_id = video['video_id']
                target_dir = video['split']  # get destination dir
                gloss_name = gloss['gloss']
                videos_dict[video_id] = (target_dir, gloss_name)
        else:
            break

    return videos_dict


def save_in_dirs(path_dataset, path_k_glosses_dir, videos):
    print('\n[log] > Copying videos in their own dir ...')
    # copy the videos in dirs
    for video_id, data in tqdm(videos.items()):
        source_url = path_dataset + video_id + '.mp4'
        destination_url = path_k_glosses_dir + data[0] + '/' + data[1] + '/'
        shutil.copy(source_url, destination_url)


def make_target_dirs(json_file, glosses, path_k_glosses_dir):
    # delete the existing target dir, if it exists
    if os.path.isdir('./' + path_k_glosses_dir):
        shutil.rmtree(path_k_glosses_dir)

    # create the target dir
    os.mkdir(path_k_glosses_dir)
    # create the train, val and test dirs
    os.mkdir(path_k_glosses_dir + 'train')
    os.mkdir(path_k_glosses_dir + 'val')
    os.mkdir(path_k_glosses_dir + 'test')

    print('\n[log] > Creating dirs ...')
    for k, gloss in tqdm(enumerate(json_file)):  # iterate through each gloss
        if k < glosses:
            # create as many folders as there are glosses
            os.mkdir(path_k_glosses_dir + 'train/' + gloss['gloss'])
            os.mkdir(path_k_glosses_dir + 'val/' + gloss['gloss'])
            os.mkdir(path_k_glosses_dir + 'test/' + gloss['gloss'])
        else:
            break


def show_info(path_k_glosses_dir):
    # print the numbers of videos
    print_entries(path_k_glosses_dir)

    # print the videos info
    print_videos_info(path_k_glosses_dir)


def print_entries(path_root):
    path_train = path_root + 'train/'
    path_val = path_root + 'val/'
    path_test = path_root + 'test/'

    n_tot = sum([len(files) for _, _, files in os.walk(path_root)])
    n_train = sum([len(files) for _, _, files in os.walk(path_train)])
    n_val = sum([len(files) for _, _, files in os.walk(path_val)])
    n_test = sum([len(files) for _, _, files in os.walk(path_test)])

    print('\n[log] > Dataset summary:')
    print(f'Total videos: {n_tot}')
    print(f'Videos in train: {n_train} - {(n_train / n_tot * 100):,.0f}%')
    print(f'Videos in val:   {n_val} - {(n_val / n_tot * 100):,.0f}%')
    print(f'Videos in test:  {n_test} - {(n_test / n_tot * 100):,.0f}%')


def print_videos_info(path_root):
    videos = get_videos_path(path_root)
    info = get_videos_info(videos)

    print('\n[log] > Dataset info:')
    print(
        f'The video {info[0][0]} has the MIN length: {info[0][1]} - '
        f'Total frames: {info[0][2]}'
    )
    print(
        f'The video {info[-1][0]} has the MAX length: {info[-1][1]} - '
        f'Total frames: {info[-1][2]}'
    )


def get_videos_path(path_root):
    # get videos path
    paths = []
    for root, dirs, files in os.walk(os.path.relpath(path_root)):
        for file in files:
            paths.append(os.path.join(root, file))

    return paths


def get_videos_info(videos):
    print('\n[log] > Retrieving videos metadata ...')
    lengths = [get_meta_data(vid_path) for vid_path in tqdm(videos)]

    return sorted(lengths, key=lambda x: x[1])  # sorted by duration


def get_meta_data(file_path):
    video_cap = cv2.VideoCapture(file_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    video_cap.release()

    file_name = os.path.basename(os.path.normpath(file_path))
    return file_name, duration, frame_count


if __name__ == '__main__':
    main()