import glob
import yaml

thisdir = '/'.join(__file__.split('/')[:-1])

def load_preset(name="default"):
    if '/' in name:
        # if open user directly
        filename = name
    else:
        # else load built-in preset of that name
        filename = f"{thisdir}/{name}.yml"
    with open(filename, 'r') as fdata:
        try:
            yamldict = yaml.safe_load(fdata)
        except yaml.YAMLError as err:
            print(err)
    return yamldict
