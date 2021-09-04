
# a load of utility functions used by STRAUSS

def nested_dict_reassign(fromdict, todict):
    """recurse through dictionaries and sub-dictionaries"""
    for k, v in fromdict.items():
        if isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_reassign(todict[k], v)
        else:
            # reassign todict value
            todict[k] = v

def nested_dict_fill(fromdict, todict):
    """recurse through dictionaries and sub-dictionaries"""
    for k, v in fromdict.items():
        if k not in todict:
            # assign todict value
            todict[k] = v
        elif isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_fill(todict[k], v)
            
def nested_dict_idx_reassign(fromdict, todict, idx):
    """recurse through dictionaries and sub-dictionaries"""
    for k, v in fromdict.items():
        if isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_idx_reassign(todict[k], v, idx)
        else:
            # reassign todict value
            todict[k] = v[idx]

            
def const_or_evo_func(x):
    """if x is callable, return x, else provide a function that returns x"""
    if callable(x):
        return x
    else:
        return lambda y: y*0 + x

def const_or_evo(x,t):
    """if x is callable, return x(t), else return x"""
    if callable(x):
        return x(t)
    else:
        return x
