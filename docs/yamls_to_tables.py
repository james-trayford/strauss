"""Generate the tables for the Parameter Reference and Styles pages.

Called from conf.py on every build. Anything going wrong here raises, so that
the build fails rather than publishing without the tables.
"""
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel

from strauss.audio_figure import _kw_defaults
from strauss.sources import (categorical, evolvable, mappable, param_lim_dict,
                             param_names)
from strauss.styles import GeneratorStyle, Mapping, Style

PRESETS = Path(__file__).resolve().parents[1] / "src" / "strauss" / "presets"
STYLES = PRESETS.parent / "styles"

# the styles shipped with strauss, in the order they appear in the docs. Listed
# rather than globbed, as a local styles directory can hold other files
BUILTIN_STYLES = ('default', 'windy', 'flute_section', 'mallets', 'sci_fi',
                  'twinkle', 'stars_appearing')

# style fields with their own table, so left out of the field reference
BEHAVIOUR_OPTIONS = ('handle_nans', 'max_notes_per_sec', 'merge_mode',
                     'angle_unit', 'pitch_binning')

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


def format_type(annotation):
    origin = get_origin(annotation)
    args = [a for a in get_args(annotation) if a is not type(None)]
    if origin is Literal:
        return ', '.join(f"`{a}`" for a in args)
    if origin is Union:
        return ' or '.join(format_type(a) for a in args)
    if origin is list:
        return f"list of {format_type(args[0])}"
    if origin is tuple:
        return f"pair of {format_type(args[0])}"
    return annotation.__name__


def format_default(field):
    if field.is_required():
        return 'required'
    if field.default is None:
        return MISSING
    if isinstance(field.default, BaseModel):
        return f"default `{type(field.default).__name__}`"
    return f"`{format_range(field.default)}`"


def behaviour_table():
    rows = []
    for name in BEHAVIOUR_OPTIONS:
        field = Style.model_fields[name]
        rows.append([f"`{name}`", field.description,
                     format_type(field.annotation), format_default(field),
                     'yes' if name in _kw_defaults else 'no'])
    columns = ['Option', 'Description', 'Allowed Values', 'Default Value',
               '`sonify` Keyword']
    return table(columns, rows)


def schema_table(model, exclude=()):
    rows = [[f"`{name}`", field.description, format_type(field.annotation),
             format_default(field)]
            for name, field in model.model_fields.items()
            if name not in exclude]
    if not rows:
        raise ValueError(f"No fields found for '{model.__name__}'")
    columns = ['Parameter', 'Description', 'Type', 'Default Value']
    return f"\n# `{model.__name__}`\n\n{table(columns, rows)}"


def builtin_styles_table():
    rows = []
    for name in BUILTIN_STYLES:
        style = Style(**read_yaml(STYLES / f"{name}.yml"))
        if not style.description:
            raise ValueError(f"Style '{name}' has no description to document")
        generator = style.generator
        sound = generator.sample or generator.preset
        data = ', '.join(f"`{m.output}`" for m in style.map if m.fixed is None)
        rows.append([f"`{name}`", ' '.join(style.description.split()),
                     data, style.sources, generator.type, f"`{sound}`"])

    columns = ['Style', 'Description', 'Data Maps To (In Order)', 'Sources',
               'Generator', 'Sound']
    return table(columns, rows)


def write_style_tables(builtin_path, behaviour_path, reference_path):
    Path(builtin_path).write_text(builtin_styles_table(), encoding='utf-8')
    Path(behaviour_path).write_text(behaviour_table(), encoding='utf-8')
    tables = [schema_table(Style, exclude=BEHAVIOUR_OPTIONS),
              schema_table(GeneratorStyle), schema_table(Mapping)]
    Path(reference_path).write_text('\n'.join(tables), encoding='utf-8')
