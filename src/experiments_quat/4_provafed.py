import json

data = {}
data['global_parameters'] = {'experiment_description': 'test experiment',
                     'dataset': 'iemocap',
                     'num_experiment': 0,
                     'num_folds': 1,
                     'num_epochs': 3,
                     'learning_rate': 0.00005,
                     'batch_size': 10}
data[1] = {'comment_1': "prova1",
                 'comment_2': "prova1"
                 }
data[2] = {'comment_1': "prova2",
                 'comment_2': "prova2"
                 }

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)
