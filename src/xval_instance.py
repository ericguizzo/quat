from __future__ import print_function
import numpy as np
import os, sys
import subprocess
import time
import shutil
import loadconfig
import configparser
'''
Crossvalidation instance script.
Automatically manage train/val/test split, call training_torch.py in a subprocess
for every fold, organize all results in a ditionary
'''
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

BACKEND = cfg.get('backend', 'backend')

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


def run_experiment(num_experiment, num_run, num_folds, dataset, experiment_folder, parameters, gpu_ID, task_type, generator):
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
        unrolled = parameters.split('/')
        for param in unrolled:
            if 'task_type' in param:
                exec(param)
        #create paths
        num_fold = i

        #init paths
        model_name = output_models_path + '/model_xval_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold)
        results_name = output_temp_results_path + '/temp_results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold) + '.npy'
        parameters_name = output_parameters_path + '/parameters_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) +  '.txt'

        #init results as ERROR
        np.save(results_name, np.array(['ERROR']))

        #run training
        time_start = time.clock()
        if BACKEND == 'keras':
            training = subprocess.Popen(['python3', 'training_keras.py',
                                         'crossvalidation', str(num_experiment), str(num_run),
                                          str(num_fold), parameters, model_name, results_name,
                                          output_temp_data_path, dataset, str(gpu_ID),
                                          str(num_folds), locals()['task_type'],parameters_name,
                                          task_type, str(generator)])
        if BACKEND == 'torch':
            training = subprocess.Popen(['python3', 'training_torch.py',
                                         'crossvalidation', str(num_experiment), str(num_run),
                                          str(num_fold), parameters, model_name, results_name,
                                          output_temp_data_path, dataset, str(gpu_ID),
                                          str(num_folds), locals()['task_type'],parameters_name,
                                          task_type, str(generator)])
        training.communicate()
        training.wait()

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


    folds['summary'] = {'training':{'mean_loss': tr_mean,
                                    'loss_std': tr_std},
                        'validation':{'mean_loss': val_mean,
                                    'loss_std': val_std},
                        'test':{'mean_loss': test_mean,
                                    'loss_std': test_std}}

    folds['summary']['training']['mean_loss_valence'] = tr_mean_valence
    folds['summary']['training']['loss_std_valence'] = tr_std_valence
    folds['summary']['validation']['mean_loss_valence'] = val_mean_valence
    folds['summary']['validation']['loss_std_valence'] = val_std_valence
    folds['summary']['test']['mean_loss_valence'] = test_mean_valence
    folds['summary']['test']['loss_std_valence'] = test_std_valence

    folds['summary']['training']['mean_loss_arousal'] = tr_mean_arousal
    folds['summary']['training']['loss_std_arousal'] = tr_std_arousal
    folds['summary']['validation']['mean_loss_arousal'] = val_mean_arousal
    folds['summary']['validation']['loss_std_arousal'] = val_std_arousal
    folds['summary']['test']['mean_loss_arousal'] = test_mean_arousal
    folds['summary']['test']['loss_std_arousal'] = test_std_arousal

    folds['summary']['training']['mean_loss_dominance'] = tr_mean_dominance
    folds['summary']['training']['loss_std_dominance'] = tr_std_dominance
    folds['summary']['validation']['mean_loss_dominance'] = val_mean_dominance
    folds['summary']['validation']['loss_std_dominance'] = val_std_dominance
    folds['summary']['test']['mean_loss_dominance'] = test_mean_dominance
    folds['summary']['test']['loss_std__dominance'] = test_std_dominance


    #compute mean acc and acc std if task_type is classification
    if locals()['task_type'] == 'regression':
        tr_RMSE = []
        val_RMSE = []
        test_RMSE = []
        tr_MAE = []
        val_MAE = []
        test_MAE = []

        tr_RMSE_valence = []
        val_RMSE_valence = []
        test_RMSE_valence = []
        tr_MAE_valence = []
        val_MAE_valence = []
        test_MAE_valence = []

        tr_RMSE_arousal = []
        val_RMSE_arousal = []
        test_RMSE_arousal = []
        tr_MAE_arousal = []
        val_MAE_arousal = []
        test_MAE_arousal = []

        tr_RMSE_dominance = []
        val_RMSE_dominance = []
        test_RMSE_dominance = []
        tr_MAE_dominance = []
        val_MAE_dominance = []
        test_MAE_dominance = []


        for i in range(num_folds):
            tr_RMSE.append(folds[i]['train_RMSE'])
            val_RMSE.append(folds[i]['val_RMSE'])
            test_RMSE.append(folds[i]['test_RMSE'])
            tr_MAE.append(folds[i]['train_MAE'])
            val_MAE.append(folds[i]['val_MAE'])
            test_MAE.append(folds[i]['test_MAE'])

            tr_RMSE_valence.append(folds[i]['train_RMSE_valence'])
            val_RMSE_valence.append(folds[i]['val_RMSE_valence'])
            test_RMSE_valence.append(folds[i]['test_RMSE_valence'])
            tr_MAE_valence.append(folds[i]['train_MAE_valence'])
            val_MAE_valence.append(folds[i]['val_MAE_valence'])
            test_MAE_valence.append(folds[i]['test_MAE_valence'])

            tr_RMSE_arousal.append(folds[i]['train_RMSE_arousal'])
            val_RMSE_arousal.append(folds[i]['val_RMSE_arousal'])
            test_RMSE_arousal.append(folds[i]['test_RMSE_arousal'])
            tr_MAE_arousal.append(folds[i]['train_MAE_arousal'])
            val_MAE_arousal.append(folds[i]['val_MAE_arousal'])
            test_MAE_arousal.append(folds[i]['test_MAE_arousal'])

            tr_RMSE_dominance.append(folds[i]['train_RMSE_dominance'])
            val_RMSE_dominance.append(folds[i]['val_RMSE_dominance'])
            test_RMSE_dominance.append(folds[i]['test_RMSE_dominance'])
            tr_MAE_dominance.append(folds[i]['train_MAE_dominance'])
            val_MAE_dominance.append(folds[i]['val_MAE_dominance'])
            test_MAE_dominance.append(folds[i]['test_MAE_dominance'])


        tr_mean_RMSE = np.mean(tr_RMSE)
        val_mean_RMSE = np.mean(val_RMSE)
        test_mean_RMSE = np.mean(test_RMSE)
        tr_std_RMSE = np.std(tr_RMSE)
        val_std_RMSE = np.std(val_RMSE)
        test_std_RMSE = np.std(test_RMSE)
        tr_mean_MAE = np.mean(tr_MAE)
        val_mean_MAE = np.mean(val_MAE)
        test_mean_MAE = np.mean(test_MAE)
        tr_std_MAE = np.std(tr_MAE)
        val_std_MAE = np.std(val_MAE)
        test_std_MAE = np.std(test_MAE)

        tr_mean_RMSE_valence = np.mean(tr_RMSE_valence)
        val_mean_RMSE_valence = np.mean(val_RMSE_valence)
        test_mean_RMSE_valence = np.mean(test_RMSE_valence)
        tr_std_RMSE_valence = np.std(tr_RMSE_valence)
        val_std_RMSE_valence = np.std(val_RMSE_valence)
        test_std_RMSE_valence = np.std(test_RMSE_valence)
        tr_mean_MAE_valence = np.mean(tr_MAE_valence)
        val_mean_MAE_valence = np.mean(val_MAE_valence)
        test_mean_MAE_valence = np.mean(test_MAE_valence)
        tr_std_MAE_valence = np.std(tr_MAE_valence)
        val_std_MAE_valence = np.std(val_MAE_valence)
        test_std_MAE_valence = np.std(test_MAE_valence)

        tr_mean_RMSE_arousal = np.mean(tr_RMSE_arousal)
        val_mean_RMSE_arousal = np.mean(val_RMSE_arousal)
        test_mean_RMSE_arousal = np.mean(test_RMSE_arousal)
        tr_std_RMSE_arousal = np.std(tr_RMSE_arousal)
        val_std_RMSE_arousal = np.std(val_RMSE_arousal)
        test_std_RMSE_arousal = np.std(test_RMSE_arousal)
        tr_mean_MAE_arousal = np.mean(tr_MAE_arousal)
        val_mean_MAE_arousal = np.mean(val_MAE_arousal)
        test_mean_MAE_arousal = np.mean(test_MAE_arousal)
        tr_std_MAE_arousal = np.std(tr_MAE_arousal)
        val_std_MAE_arousal = np.std(val_MAE_arousal)
        test_std_MAE_arousal = np.std(test_MAE_arousal)

        tr_mean_RMSE_dominance = np.mean(tr_RMSE_dominance)
        val_mean_RMSE_dominance = np.mean(val_RMSE_dominance)
        test_mean_RMSE_dominance = np.mean(test_RMSE_dominance)
        tr_std_RMSE_dominance = np.std(tr_RMSE_dominance)
        val_std_RMSE_dominance = np.std(val_RMSE_dominance)
        test_std_RMSE_dominance = np.std(test_RMSE_dominance)
        tr_mean_MAE_dominance = np.mean(tr_MAE_dominance)
        val_mean_MAE_dominance = np.mean(val_MAE_dominance)
        test_mean_MAE_dominance = np.mean(test_MAE_dominance)
        tr_std_MAE_dominance = np.std(tr_MAE_dominance)
        val_std_MAE_dominance = np.std(val_MAE_dominance)
        test_std_MAE_dominance = np.std(test_MAE_dominance)


        folds['summary']['training']['mean_RMSE'] =  tr_mean_RMSE
        folds['summary']['training']['RMSE_std'] = tr_std_RMSE
        folds['summary']['validation']['mean_RMSE'] =  val_mean_RMSE
        folds['summary']['validation']['RMSE_std'] = val_std_RMSE
        folds['summary']['test']['mean_RMSE'] =  test_mean_RMSE
        folds['summary']['test']['RMSE_std'] = test_std_RMSE
        folds['summary']['training']['mean_MAE'] =  tr_mean_MAE
        folds['summary']['training']['MAE_std'] = tr_std_MAE
        folds['summary']['validation']['mean_MAE'] =  val_mean_MAE
        folds['summary']['validation']['MAE_std'] = val_std_MAE
        folds['summary']['test']['mean_MAE'] =  test_mean_MAE
        folds['summary']['test']['MAE_std'] = test_std_MAE

        folds['summary']['training']['mean_RMSE_valence'] =  tr_mean_RMSE_valence
        folds['summary']['training']['RMSE_std_valence'] = tr_std_RMSE_valence
        folds['summary']['validation']['mean_RMSE_valence'] =  val_mean_RMSE_valence
        folds['summary']['validation']['RMSE_std_valence'] = val_std_RMSE_valence
        folds['summary']['test']['mean_RMSE_valence'] =  test_mean_RMSE_valence
        folds['summary']['test']['RMSE_std_valence'] = test_std_RMSE_valence
        folds['summary']['training']['mean_MAE_valence'] =  tr_mean_MAE_valence
        folds['summary']['training']['MAE_std_valence'] = tr_std_MAE_valence
        folds['summary']['validation']['mean_MAE_valence'] =  val_mean_MAE_valence
        folds['summary']['validation']['MAE_std_valence'] = val_std_MAE_valence
        folds['summary']['test']['mean_MAE_valence'] =  test_mean_MAE_valence
        folds['summary']['test']['MAE_std_valence'] = test_std_MAE_valence

        folds['summary']['training']['mean_RMSE_arousal'] =  tr_mean_RMSE_arousal
        folds['summary']['training']['RMSE_std_arousal'] = tr_std_RMSE_arousal
        folds['summary']['validation']['mean_RMSE_arousal'] =  val_mean_RMSE_arousal
        folds['summary']['validation']['RMSE_std_arousal'] = val_std_RMSE_arousal
        folds['summary']['test']['mean_RMSE_arousal'] =  test_mean_RMSE_arousal
        folds['summary']['test']['RMSE_std_arousal'] = test_std_RMSE_arousal
        folds['summary']['training']['mean_MAE_arousal'] =  tr_mean_MAE_arousal
        folds['summary']['training']['MAE_std_arousal'] = tr_std_MAE_arousal
        folds['summary']['validation']['mean_MAE_arousal'] =  val_mean_MAE_arousal
        folds['summary']['validation']['MAE_std_arousal'] = val_std_MAE_arousal
        folds['summary']['test']['mean_MAE_arousal'] =  test_mean_MAE_arousal
        folds['summary']['test']['MAE_std_arousal'] = test_std_MAE_arousal

        folds['summary']['training']['mean_RMSE_dominance'] =  tr_mean_RMSE_dominance
        folds['summary']['training']['RMSE_std_dominance'] = tr_std_RMSE_dominance
        folds['summary']['validation']['mean_RMSE_dominance'] =  val_mean_RMSE_dominance
        folds['summary']['validation']['RMSE_std_dominance'] = val_std_RMSE_dominance
        folds['summary']['test']['mean_RMSE_dominance'] =  test_mean_RMSE_dominance
        folds['summary']['test']['RMSE_std_dominance'] = test_std_RMSE_dominance
        folds['summary']['training']['mean_MAE_dominance'] =  tr_mean_MAE_dominance
        folds['summary']['training']['MAE_std_dominance'] = tr_std_MAE_dominance
        folds['summary']['validation']['mean_MAE_dominance'] =  val_mean_MAE_dominance
        folds['summary']['validation']['MAE_std_dominance'] = val_std_MAE_dominance
        folds['summary']['test']['mean_MAE_dominance'] =  test_mean_MAE_dominance
        folds['summary']['test']['MAE_std_dominance'] = test_std_MAE_dominance
    else:
        tr_acc = []
        val_acc = []
        test_acc = []
        tr_f1 = []
        val_f1 = []
        test_f1 = []
        tr_precision = []
        val_precision = []
        test_precision = []
        tr_recall = []
        val_recall = []
        test_recall = []
        for i in range(num_folds):
            tr_acc.append(folds[i]['train_acc'])
            val_acc.append(folds[i]['val_acc'])
            test_acc.append(folds[i]['test_acc'])
            tr_f1.append(folds[i]['train_f1'])
            val_f1.append(folds[i]['val_f1'])
            test_f1.append(folds[i]['test_f1'])
            tr_precision.append(folds[i]['train_precision'])
            val_precision.append(folds[i]['val_precision'])
            test_precision.append(folds[i]['test_precision'])
            tr_recall.append(folds[i]['train_recall'])
            val_recall.append(folds[i]['val_recall'])
            test_recall.append(folds[i]['test_recall'])
        tr_mean_acc = np.mean(tr_acc)
        val_mean_acc = np.mean(val_acc)
        test_mean_acc = np.mean(test_acc)
        tr_std_acc = np.std(tr_acc)
        val_std_acc = np.std(val_acc)
        test_std_acc = np.std(test_acc)
        tr_mean_f1 = np.mean(tr_f1)
        val_mean_f1 = np.mean(val_f1)
        test_mean_f1 = np.mean(test_f1)
        tr_std_f1 = np.std(tr_f1)
        val_std_f1 = np.std(val_f1)
        test_std_f1 = np.std(test_f1)
        tr_mean_precision = np.mean(tr_precision)
        val_mean_precision = np.mean(val_precision)
        test_mean_precision = np.mean(test_precision)
        tr_std_precision = np.std(tr_precision)
        val_std_precision = np.std(val_precision)
        test_std_precision = np.std(test_precision)
        tr_mean_recall = np.mean(tr_recall)
        val_mean_recall = np.mean(val_recall)
        test_mean_recall = np.mean(test_recall)
        tr_std_recall = np.std(tr_recall)
        val_std_recall = np.std(val_recall)
        test_std_recall = np.std(test_recall)
        folds['summary']['training']['mean_acc'] =  tr_mean_acc
        folds['summary']['training']['acc_std'] = tr_std_acc
        folds['summary']['training']['mean_f1'] =  tr_mean_f1
        folds['summary']['training']['f1_std'] = tr_std_f1
        folds['summary']['training']['mean_precision'] =  tr_mean_precision
        folds['summary']['training']['precision_std'] = tr_std_precision
        folds['summary']['training']['mean_recall'] =  tr_mean_recall
        folds['summary']['training']['recall_std'] = tr_std_recall
        folds['summary']['validation']['mean_acc'] =  val_mean_acc
        folds['summary']['validation']['acc_std'] = val_std_acc
        folds['summary']['validation']['mean_f1'] =  val_mean_f1
        folds['summary']['validation']['f1_std'] = val_std_f1
        folds['summary']['validation']['mean_precision'] =  val_mean_precision
        folds['summary']['validation']['precision_std'] = val_std_precision
        folds['summary']['validation']['mean_recall'] =  val_mean_recall
        folds['summary']['validation']['recall_std'] = val_std_recall
        folds['summary']['test']['mean_acc'] =  test_mean_acc
        folds['summary']['test']['acc_std'] = test_std_acc
        folds['summary']['test']['mean_f1'] =  test_mean_f1
        folds['summary']['test']['f1_std'] = test_std_f1
        folds['summary']['test']['mean_precision'] =  test_mean_precision
        folds['summary']['test']['precision_std'] = test_std_precision
        folds['summary']['test']['mean_recall'] =  test_mean_recall
        folds['summary']['test']['recall_std'] = test_std_recall


    folds['summary']['parameters'] = parameters
    print ('\n Results summary:')
    print (folds['summary'])
    print ('')
    print ('')
    print ('\n CROSSVALIDATION COMPLETED')
    print ('')
    print ('')


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
