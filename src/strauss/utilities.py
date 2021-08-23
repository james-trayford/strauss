
# a load of utility functions used by STRAUSS

def nested_dict_reassign(original, new):
    """recurse through dictionaries and sub-dictionaries"""
    for k, v in new.items():
        if isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_reassign(original[k], v)
        else:
            # reassign original value
            original[k] = v

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
