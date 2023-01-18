import yaml
import os 

def load_config(config_file_path):
    '''receives a config file path and loads into a json object'''
    try:
        config = yaml.safe_load(open(config_file_path,'r'))
        config['spec']['min_hz'] = None if config['spec']['min_hz'] == 'None' else config['spec']['min_hz']
        config['spec']['max_hz'] = None if config['spec']['max_hz'] == 'None' else config['spec']['max_hz']
        config['spec']['db_limits'] = (config['spec']['min_db'], config['spec']['max_db'])
    except Exception as e:
        print(f'Could not load file {config_file_path}')
        print(e)
        
    else:
        try:
            os.makedirs(config['outpath'])
        except FileExistsError as fe:
            print('WARNING: output directory already exists')
        else:
            print(f"An output file was created at '{config['outpath']}'")
        return config