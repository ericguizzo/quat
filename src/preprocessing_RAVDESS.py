from __future__ import print_function
import configparser
import utility_functions as uf
import preprocessing_utils as pre
import numpy as np
import random
import os, sys
'''
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
#in
INPUT_RAVDESS_FOLDER =  cfg.get('preprocessing', 'input_audio_folder_ravdess')
#out
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))
'''

cfg = configparser.ConfigParser()
cfg.read('preprocessing_config.ini')

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
AUGMENTATION = eval(cfg.get('feature_extraction', 'augmentation'))
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
SEGMENTATION = False
INPUT_RAVDESS_FOLDER = cfg.get('preprocessing', 'input_ravdess_folder')
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')
FIXED_SEED = cfg.getint('sampling', 'fixed_seed')

if FIXED_SEED is not None:
    # Set seed
    manualSeed = FIXED_SEED
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    seed=manualSeed
    np.random.seed(seed)

#number of classes of current dataset
num_classes_ravdess = 8

def get_max_length_RAVDESS(input_folder=INPUT_RAVDESS_FOLDER, sr=SR):
    '''
    get longest audio file (insamples) for eventual zeropadding
    '''
    max_file_length, sr = uf.find_longest_audio(input_folder)
    max_file_length = int(max_file_length * sr)

    return max_file_length

def get_label_RAVDESS(input_soundfile, num_classes=num_classes_ravdess):
    '''
    compute label starting from soundfile
    '''
    label = input_soundfile.split('/')[-1].split('.')[0].split('-')[2]
    one_hot_label = (uf.onehot(int(label)-1, num_classes_ravdess))

    return one_hot_label


def filter_data_RAVDESS(contents, criterion, item_to_filter):
    '''
    outputs list of soundfiles of the only actor number "item_to_filter"
    '''
    actor = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-1])
    statement = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-3])
    intensity = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-4])
    gender = lambda x: 1 if int(x.split('/')[-1].split('.')[0].split('-')[-1]) % 2. == 0 else 0

    if criterion == 'actor':
        filtered = list(filter(lambda x: actor(x) in item_to_filter, contents))
    elif criterion == 'statement':
        filtered = list(filter(lambda x: statement(x) in item_to_filter, contents))
    elif criterion == 'intensity':
        filtered = list(filter(lambda x: intensity(x) in item_to_filter, contents))
    elif criterion == 'gender':
        filtered = list(filter(lambda x: gender(x) in item_to_filter, contents))

    return filtered


def main():
    '''
    custom preprocessing routine for the ravdess dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    #compute max file length of current dataet
    #for the zeropadding
    #max_file_length = get_max_length_RAVDESS()
    max_file_length = SR * 4
    #define the list of foldable items. In the case of RAVDESS
    #actors are simply numbered from 0 to 24
    contents = os.listdir(INPUT_RAVDESS_FOLDER)  #get list of filepaths
    contents = list(filter(lambda x: '.wav' in x, contents))  #keep only wav files
    random.shuffle(contents)
    #actors_list = actors_list[:2]
    num_files = len(contents)
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    predictors_save_path = os.path.join(OUTPUT_FOLDER, 'ravdess_randsplit' + appendix + '_predictors.npy')
    target_save_path = os.path.join(OUTPUT_FOLDER, 'ravdess_randsplit' + appendix + '_target.npy')
    #iterate the list of actors
    index = 1  #index for progress bar
    for i in contents:
        #print progress bar
        #uf.print_bar(index, num_files)
        #get only soundpaths of current actor
        curr_list = [i]
        #fold_string = '\nPreprocessing foldable item: ' + str(index) + '/' + str(num_files)
        #print (fold_string)
        #print ('\nPreprocessing items')
        #make sure that each item list is a FULL path to a sound file
        #and not only the sound name as os.listdir outputs
        curr_list = [os.path.join(INPUT_RAVDESS_FOLDER, x) for x in curr_list]
        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath
        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_RAVDESS)
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
