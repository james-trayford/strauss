import glob
import yaml

thisdir = '/'.join(__file__.split('/')[:-1])

def load_preset(name="default"):
    filename = f"{thisdir}/{name}.yml"
    with open(filename, 'r') as fdata:
        try:
            yamldict = yaml.safe_load(fdata)
        except yaml.YAMLError as err:
            print(err)
    return yamldict
