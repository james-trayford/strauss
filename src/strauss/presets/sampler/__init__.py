import glob
import yaml
import pathlib

##thisdir = '/'.join(__file__.split('/')[:-1])
path = pathlib.PurePath(__file__)
thisdir = path.parent

def read_yaml(filename):
    ##with open(filename, 'r') as fdata:
    with filename.open(mode='r') as fdata:
        try:
            yamldict = yaml.safe_load(fdata)
        except yaml.YAMLError as err:
            print(err)
    return yamldict

def load_ranges(name="default"):
    ##filename = f"{thisdir}/ranges/default.yml"
    filename = pathlib.PurePath(f"{thisdir}","ranges","default.yml")
    return read_yaml(filename)

def load_preset(name="default"):
    ##if '/' in name:
    if pathlib.PurePath(name).name != pathlib.PurePath(name):
        # if open user directly
        filename = pathlib.PurePath(name)
    else:
        # else load built-in preset of that name
        filename = pathlib.PurePath(f"{thisdir}", f"{name}.yml")
    return read_yaml(filename)

def preset_details(name='*'):
    pres = glob.glob(pathlib.PurePath(f"{thisdir}", f"*{name.lower()}*.yml"))
    for p in pres:
        with p.open(mode='r') as fdata:
            try:
                d = yaml.safe_load(fdata)
                print(f"\033[1m{d['name']}:\033[0m\n{d['description']}\n")
            except:
                pass
