import yaml

def get_config(configfile):
    return yaml.safe_load(open(configfile, 'r'))
    