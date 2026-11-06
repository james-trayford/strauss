"""Generate docs/tables.md, the parameter tables for the Parameter Reference.

Called from conf.py on every build. Anything going wrong here raises, so that
the build fails rather than publishing without the tables.
"""
from pathlib import Path

import yaml

from strauss.sources import (categorical, evolvable, mappable, param_lim_dict,
                             param_names)

PRESETS = Path(__file__).resolve().parents[1] / "src" / "strauss" / "presets"

# in the order they appear in the docs
GENERATORS = {'synth': "`Synthesizer` Generator",
              'sampler': "`Sampler` Generator",
              'spec': "`Spectralizer` Generator",
              'speech': "`Speech` Generator"}

# aliases of 'azimuth' and 'polar', described in the text of the page
ALIASES = ('phi', 'theta')

# mappable parameters set by sources alone, so without an entry in the ranges
SOURCE_UNITS = {'time': 'fraction of duration',
                'time_evo': 'fraction of duration'}

MISSING = '-'


def read_yaml(filename):
    with filename.open(mode='r') as fdata:
        return yaml.safe_load(fdata)


def table(columns, rows):
    head = f"| {' | '.join(columns)} |\n"
    rule = f"| {' | '.join(['-----------'] * len(columns))} |\n"
    return head + rule + ''.join(f"| {' | '.join(r)} |\n" for r in rows)


def format_number(value):
    # YAML reads values like 1e-2 and 2.2e4 as strings
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def format_range(value):
    if isinstance(value, (list, tuple)):
        return ' to '.join(format_number(v) for v in value)
    return format_number(value)


def preset_tables(metadict, valdict, rdict, headlev):
    """Tables for one level of a preset and, as subsections, those below it."""
    intro = ''
    rows = []
    sections = ''

    for k, meta in metadict.items():
        if k == '_doc':
            intro = f"\n{meta}\n"
        elif hasattr(meta, 'keys'):
            sections += f"\n{'#' * headlev} `{k}` parameter group\n"
            sections += preset_tables(meta, valdict[k], rdict.get(k, {}),
                                      headlev + 1)
        else:
            limits = format_range(rdict[k]) if k in rdict else MISSING
            rows.append([f"`{k}`", str(meta).strip(), str(valdict[k]),
                         f"`{limits}`", rdict.get(f"{k}_unit", MISSING)])

    columns = ['Parameter', 'Description', 'Default Value', 'Default Range',
               'Unit']
    rendered = f"\n{table(columns, rows)}" if rows else ''
    return f"{intro}{rendered}{sections}"


def lookup_unit(ranges, key):
    """The unit a mappable parameter is given by the first ranges declaring it."""
    if key in SOURCE_UNITS:
        return SOURCE_UNITS[key]

    *groups, name = key.split('/')
    for rdict in ranges:
        for group in groups:
            rdict = rdict.get(group, {})
        if f"{name}_unit" in rdict:
            return rdict[f"{name}_unit"]
    return MISSING


def mappable_table(ranges):
    rows = []
    for key in mappable:
        if key in ALIASES:
            continue
        if key in categorical:
            limits = MISSING
        else:
            limits = format_range(param_lim_dict[key])
        rows.append([f"`{key}`", param_names[key],
                     'yes' if key in evolvable else 'no',
                     f"`{limits}`", lookup_unit(ranges, key)])

    columns = ['Parameter', 'Name', 'Evolvable', 'Default Limits', 'Unit']
    return f"\n# Mappable Parameters\n\n{table(columns, rows)}"


def write_tables(path):
    presets = {}
    ranges = []
    for gen in GENERATORS:
        presets[gen] = read_yaml(PRESETS / gen / "default.yml")
        ranges.append(read_yaml(PRESETS / gen / "ranges" / "default.yml"))

    sections = [mappable_table(ranges)]
    for (gen, title), rdict in zip(GENERATORS.items(), ranges):
        body = preset_tables(presets[gen]['_meta'], presets[gen], rdict, 2)
        if '| Parameter |' not in body:
            raise ValueError(f"No parameter table generated for '{gen}'")
        sections.append(f"\n# {title}\n{body}")

    Path(path).write_text('\n---\n'.join(sections), encoding='utf-8')
