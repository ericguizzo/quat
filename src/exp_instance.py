from __future__ import print_function
import numpy as np
import os, sys
import subprocess
import time
import shutil

def gen_command(script, d):
    command = ['python3', script]
    for i in d:
        name = '--' + i + ' '
        value = d[i]
        opt = name + str(value)
        command.append(opt)
    command = ' '.join(command)

    return command

def save_code(output_code_path):
    curr_src_path = './'
    curr_config_path = '../config/'
    output_src_path = output_code_path + '/src'
    output_config_path = output_code_path + '/config'
    line1 = 'cp ' + curr_src_path + '* ' + output_src_path
    line2 = 'cp ' + curr_config_path + '* ' + output_config_path
    copy1 = subprocess.Popen(line1, shell=True)
    copy1.communicate()
    copy2 = subprocess.Popen(line2, shell=True)
    copy2.communicate()


def run_experiment(num_experiment=0, num_run=0, num_folds=1,
                   dataset='iemocap_randsplit', experiment_folder='../temp/',
                   script='training_autoencoder.py', parameters={}):
    '''
    run the crossvalidation
    '''
    print("NEW EXPERIMENT: exp: " + str(num_experiment) + ' run: ' + str(num_run))
    print('Dataset: ' + dataset)

    #create output path if not existing
    output_path = experiment_folder + '/experiment_' + str(num_experiment)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_temp_path = output_path + '/temp'
    if not os.path.exists(output_temp_path):
        os.makedirs(output_temp_path)

    output_models_path = output_path + '/models'
    if not os.path.exists(output_models_path):
        os.makedirs(output_models_path)

    output_results_path = output_path + '/results'
    if not os.path.exists(output_results_path):
        os.makedirs(output_results_path)

    output_parameters_path = output_results_path + '/parameters'
    if not os.path.exists(output_parameters_path):
        os.makedirs(output_parameters_path)

    output_temp_data_path = output_temp_path + '/temp_data'
    if not os.path.exists(output_temp_data_path):
        os.makedirs(output_temp_data_path)

    output_temp_results_path = output_temp_path + '/temp_results'
    if not os.path.exists(output_temp_results_path):
        os.makedirs(output_temp_results_path)

    output_code_path = output_path + '/code'
    if not os.path.exists(output_code_path):
        os.makedirs(output_code_path)

    output_src_path = output_code_path + '/src'
    if not os.path.exists(output_src_path):
        os.makedirs(output_src_path)

    output_config_path = output_code_path + '/config'
    if not os.path.exists(output_config_path):
        os.makedirs(output_config_path)


    #initialize results dict
    folds = {}

    #iterate folds
    for i in range(num_folds):
        #unroll parameters to find task_type:
        #create paths
        num_fold = i

        #init paths
        model_name = output_models_path + '/model_xval_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold)
        results_name = output_temp_results_path + '/temp_results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold) + '.npy'
        parameters_name = output_parameters_path + '/parameters_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) +  '.txt'

        #init results as ERROR
        np.save(results_name, np.array(['ERROR']))

        #run training
        time_start = time.perf_counter()

        parameters['num_fold'] = num_fold
        parameters['model_path'] = model_name
        parameters['results_path'] = results_name

        shell_command = gen_command(script, parameters)

        print (shell_command)
        training = subprocess.Popen(shell_command, shell=True)


        '''
        training = subprocess.Popen(['python3', 'training_torch.py',
                                     'crossvalidation', str(num_experiment), str(num_run),
                                      str(num_fold), parameters, model_name, results_name,
                                      output_temp_data_path, dataset, str(gpu_ID),
                                      str(num_folds), locals()['task_type'],parameters_name,
                                      task_type)
        training.communicate()
        training.wait()
        '''
        sys.exit(0)

        training_time = (time.clock() - time_start)
        print ('training time: ' + str(training_time))

        #wait for file to be created
        flag = 'ERROR'
        while flag == 'ERROR':
            time.sleep(0.2)
            flag = np.load(results_name, allow_pickle=True)

        #update results dict
        temp_results = np.load(results_name, allow_pickle=True)
        temp_results = temp_results.item()
        folds[i] = temp_results
        #stop fold iter

    #compute summary
    #compute mean loss and loss std
    tr_loss = []
    val_loss = []
    test_loss = []

    tr_loss_valence = []
    val_loss_valence = []
    test_loss_valence = []

    tr_loss_arousal = []
    val_loss_arousal = []
    test_loss_arousal = []

    tr_loss_dominance = []
    val_loss_dominance = []
    test_loss_dominance = []

    for i in range(num_folds):
        tr_loss.append(folds[i]['train_loss'])
        val_loss.append(folds[i]['val_loss'])
        test_loss.append(folds[i]['test_loss'])

        tr_loss_valence.append(folds[i]['train_loss_valence'])
        val_loss_valence.append(folds[i]['val_loss_valence'])
        test_loss_valence.append(folds[i]['test_loss_valence'])

        tr_loss_arousal.append(folds[i]['train_loss_arousal'])
        val_loss_arousal.append(folds[i]['val_loss_arousal'])
        test_loss_arousal.append(folds[i]['test_loss_arousal'])

        tr_loss_dominance.append(folds[i]['train_loss_dominance'])
        val_loss_dominance.append(folds[i]['val_loss_dominance'])
        test_loss_dominance.append(folds[i]['test_loss_dominance'])

    tr_mean = np.mean(tr_loss)
    val_mean = np.mean(val_loss)
    test_mean = np.mean(test_loss)
    tr_std = np.std(tr_loss)
    val_std = np.std(val_loss)
    test_std = np.std(test_loss)

    tr_mean_valence = np.mean(tr_loss_valence)
    val_mean_valence = np.mean(val_loss_valence)
    test_mean_valence = np.mean(test_loss_valence)
    tr_std_valence = np.std(tr_loss_valence)
    val_std_valence = np.std(val_loss_valence)
    test_std_valence = np.std(test_loss_valence)

    tr_mean_arousal = np.mean(tr_loss_arousal)
    val_mean_arousal = np.mean(val_loss_arousal)
    test_mean_arousal = np.mean(test_loss_arousal)
    tr_std_arousal = np.std(tr_loss_arousal)
    val_std_arousal = np.std(val_loss_arousal)
    test_std_arousal = np.std(test_loss_arousal)

    tr_mean_dominance = np.mean(tr_loss_dominance)
    val_mean_dominance = np.mean(val_loss_dominance)
    test_mean_dominance = np.mean(test_loss_dominance)
    tr_std_dominance = np.std(tr_loss_dominance)
    val_std_dominance = np.std(val_loss_dominance)
    test_std_dominance = np.std(test_loss_dominance)


    #save results dict
    dict_name = 'results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    final_dict_path = output_results_path + '/' + dict_name
    np.save(final_dict_path, folds)

    #run training
    spreadsheet_name = dataset + '_exp' + str(num_experiment) + '_results_spreadsheet.xls'
    gen_spreadsheet = subprocess.Popen(['python3', 'results_to_excel.py',
                                        output_results_path, spreadsheet_name])
    gen_spreadsheet.communicate()
    gen_spreadsheet.wait()

    #save current code
    save_code(output_code_path)


if __name__ == '__main__':
    run_experiment()
