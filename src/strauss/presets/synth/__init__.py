import yaml
from pathlib import Path

p = Path(__file__)
thisdir = p.parent

def read_yaml(filename):
    with filename.open(mode='r') as fdata:
        try:
            yamldict = yaml.safe_load(fdata)
        except yaml.YAMLError as err:
            print(err)
    return yamldict

def load_ranges(name="default"):
    filename = Path(f"{thisdir}","ranges",f"{name}.yml")
    return read_yaml(filename)

def load_preset(name="default"):

    path = Path(name)

    # Full or relative path to an existing YAML file
    if path.suffix == ".yml":
        filename = path

    # Just a preset name
    elif path.name == str(path):
        filename = Path(f"{thisdir}", f"{path}.yml")

    # Path without extension
    else:
        filename = path.with_suffix(".yml")

    return read_yaml(filename)

def preset_details(name='*'):
    pres = sorted(Path(f"{thisdir}").glob(f"*{name.lower()}*.yml"))
    for p in pres:
        with p.open(mode='r') as fdata:
            try:
                d = yaml.safe_load(fdata)
                print(f"\033[1m{d['name']}:\033[0m\n{d['description']}\n")
            except:
                pass
