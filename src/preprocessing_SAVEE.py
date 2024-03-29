from __future__ import print_function
import configparser
import utility_functions as uf
import preprocessing_utils as pre
import numpy as np
import random
import os, sys


cfg = configparser.ConfigParser()
cfg.read('preprocessing_config.ini')

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
AUGMENTATION = eval(cfg.get('feature_extraction', 'augmentation'))
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
SEGMENTATION = False
INPUT_SAVEE_FOLDER = cfg.get('preprocessing', 'input_savee_folder')
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')
FIXED_SEED = cfg.getint('sampling', 'fixed_seed')

if FIXED_SEED is not None:
    # Set seed
    manualSeed = FIXED_SEED
    random.seed(manualSeed)
    seed=manualSeed
    np.random.seed(seed)

#number of classes of current dataset
num_classes_savee = 7

assoc_labels = {
                'a': 0,
                'd': 1,
                'f': 2,
                'h': 3,
                'n': 4,
                'sa': 5,
                'su': 6
                }

def get_label_SAVEE(input_soundfile, num_classes=num_classes_savee):
    '''
    compute label starting from soundfile
    '''
    label = os.path.basename(input_soundfile).split('.')[0][:-2]
    label_int = assoc_labels[label]
    one_hot_label = (uf.onehot(int(label_int), num_classes_savee))

    return one_hot_label

def get_all_paths(input_folder):
    '''
    get all dataset sound paths
    '''
    paths = []
    folders = os.listdir(input_folder)
    folders = [os.path.join(input_folder, i) for i in folders]
    folders = list(filter(lambda x: "DS_Store" not in x, folders))
    for f in folders:
        contents = os.listdir(f)
        contents = [os.path.join(f, i) for i in contents]
        contents = list(filter(lambda x: x[-3:] == "wav", contents))
        paths += contents

    return paths

def main():
    '''
    custom preprocessing routine for the ravdess dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')

    max_file_length = SR * 4
    contents = get_all_paths(INPUT_SAVEE_FOLDER)
    random.shuffle(contents)
    num_files = len(contents)
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    predictors_save_path = os.path.join(OUTPUT_FOLDER, 'savee_randsplit' + appendix + '_predictors.npy')
    target_save_path = os.path.join(OUTPUT_FOLDER, 'savee_randsplit' + appendix + '_target.npy')
    #iterate the list of actors
    index = 1  #index for progress bar
    for i in contents:

        curr_list = [i]
        #fold_string = '\nPreprocessing foldable item: ' + str(index) + '/' + str(num_files)
        #print (fold_string)
        #print ('\nPreprocessing items')
        #make sure that each item list is a FULL path to a sound file
        #and not only the sound name as os.listdir outputs
        #curr_list = [os.path.join(INPUT_EMOVO_FOLDER, x) for x in curr_list]
        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath
        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_SAVEE)
        uf.print_bar(index, num_files)

        #append preprocessed predictors and target to the dict
        if len(curr_predictors.shape) > 1:
            print ('COGLIONE', curr_predictors.shape)
            predictors[i] = curr_predictors
            target[i] = curr_target
            index +=1

    #save dicts
    #save dicts
    print ('\nSaving matrices...')
    np.save(predictors_save_path, predictors)
    np.save(target_save_path, target)
    #print dimensions
    count = 0
    predictors_dims = 0
    keys = list(predictors.keys())
    for i in keys:
        count += predictors[i].shape[0]
    pred_shape = np.array(predictors[keys[0]]).shape[1:]
    tg_shape = np.array(target[keys[0]]).shape[1:]
    print ('')
    print ('MATRICES SUCCESFULLY COMPUTED')
    print ('')
    print ('Total number of datapoints: ' + str(count))
    print (' Predictors shape: ' + str(pred_shape))
    print (' Target shape: ' + str(tg_shape))


if __name__ == '__main__':
    main()
