import glob
import yaml

thisdir = '/'.join(__file__.split('/')[:-1])

def read_yaml(filename):
    with open(filename, 'r') as fdata:
        try:
            yamldict = yaml.safe_load(fdata)
        except yaml.YAMLError as err:
            print(err)
    return yamldict

def load_ranges(name="default"):
    filename = f"{thisdir}/ranges/default.yml"
    return read_yaml(filename)

def load_preset(name="default"):
    if '/' in name:
        # if open user directly
        filename = name
    else:
        # else load built-in preset of that name
        filename = f"{thisdir}/{name}.yml"
    return read_yaml(filename)

def preset_details(name='*'):
    pres = glob.glob(f"{thisdir}/*{name.lower()}*.yml")
    for p in pres:
        with open(p, 'r') as fdata:
            try:
                d = yaml.safe_load(fdata)
                print(f"\033[1m{d['name']}:\033[0m\n{d['description']}\n")
            except:
                pass
