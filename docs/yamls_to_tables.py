# --- hide: start ---
import yaml
import os
from glob import glob
from pathlib import Path


generators = {'spec' : "`Spectraliser` Generator",
              'synth' : "`Synthesiser` Generator",
              'sampler' : "`Sampler` Generator"}

p = Path("src", "strauss", "presets", "*", "default.yml")

def read_yaml(filename):
    with filename.open(mode='r') as fdata:
        # try:
            yamldict = yaml.safe_load(fdata)
        # except yaml.YAMLError as err:
        #     print(err)
    return yamldict

tstr1 = "\n| Parameter | Description | Default Value | Default Range | Unit |\n"
tstr2 = "| ----------- | ----------- | ----------- | ----------- | ----------- |\n"

def yaml_traverse(metadict, valdict, rdict, headlev=1):
    if hasattr(metadict, 'keys'):
        starttab = 1
        topstr = ''
        tabstr = ''
        secstr = ''
                
        for k in metadict.keys():
            # print (f">>>>>> {k}")
            if hasattr(metadict[k], 'keys'):
                secstr += '\n'+''.join(['#']*headlev) + f" `{k}` parameter group\n"
                if not k in rdict:
                    rdict[k] = {}
                secstr += yaml_traverse(metadict[k], valdict[k], rdict[k], headlev+1)
                continue
            if k == '_doc':
                topstr += f"\n{metadict[k]}\n"
                # print('\n',metadict[k])
                continue
            if k not in rdict:
                # unspecified => '-'
                rdict[k] = '-'
            else:
                # lets avoid these line-breaking with special characters
                rdict[k] = str(rdict[k]).replace(" ","").replace(",","\u2011").replace("-","\u2011")
            if k+"_unit" not in rdict:
                # unspecified => '-'
                try:
                    float(valdict[k])
                    if not isinstance(valdict[k], bool):
                        rdict[k+"_unit"] = 'unitless'
                    else:
                        rdict[k+"_unit"] = '-'
                except ValueError:
                    rdict[k+"_unit"] = '-'
            # print(f"|`{k}` |  _{metadict[k]}_ | `{str(valdict[k]).strip()}` | `{rdict[k]}` | {rdict[k+'_unit']}")
            if starttab:     
                tabstr = tstr1 + tstr2 + tabstr
                starttab = 0
            tabstr += f"| `{k}` | {str(metadict[k]).strip()} | {valdict[k]} | `{rdict[k]}` | {rdict[k+'_unit']}\n"
            if k not in rdict:
                rdict[k] = {}
        return f'{topstr}{tabstr}{secstr}'
    
    else:
        return

with open('docs/tables.md', 'w') as outfile:
    l = len(glob(str(p)))
    i = 0
    for f in glob(str(p)):
        p = Path(f)
        ydat = read_yaml(p)
        rdat = read_yaml(p.parents[0] / "ranges" / "default.yml")
        # print(f"\n# {generators[p.parents[0].name]}\n")
        ystr = yaml_traverse(ydat['_meta'], ydat, rdat, 2)
        # print(ystr)
        # print('---')
        outfile.write(f"\n# {generators[p.parents[0].name]}\n")
        outfile.write(ystr)
        if i < l-1:
            outfile.write('---')
            i += 1

# --- hide: stop ---
