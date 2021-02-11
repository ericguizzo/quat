import argparse

keys = ['train_merda', 'train_gggg', 'val_pipipip']

train_keys = [i for i in keys if 'train' in i]


options = {'batch_size': 5, 'exp_name': 'prova'}
script = 'training_autoencoder.py'
def gen_command(script, d):
    command = ['python3', script]
    for i in d:
        name = '--' + i + ' '
        value = d[i]
        opt = name + str(value)
        command.append(opt)
    command = ' '.join(command)

    return command
    print (command)



gen_command(script,options)
