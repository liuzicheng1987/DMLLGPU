# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.11
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_DMLLGPUCpp', [dirname(__file__)])
        except ImportError:
            import _DMLLGPUCpp
            return _DMLLGPUCpp
        if fp is not None:
            try:
                _mod = imp.load_module('_DMLLGPUCpp', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _DMLLGPUCpp = swig_import_helper()
    del swig_import_helper
else:
    import _DMLLGPUCpp
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class LossFunctionCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, LossFunctionCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, LossFunctionCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DMLLGPUCpp.new_LossFunctionCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_LossFunctionCpp
    __del__ = lambda self : None;
LossFunctionCpp_swigregister = _DMLLGPUCpp.LossFunctionCpp_swigregister
LossFunctionCpp_swigregister(LossFunctionCpp)

class SquareLossCpp(LossFunctionCpp):
    __swig_setmethods__ = {}
    for _s in [LossFunctionCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SquareLossCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [LossFunctionCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, SquareLossCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DMLLGPUCpp.new_SquareLossCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_SquareLossCpp
    __del__ = lambda self : None;
SquareLossCpp_swigregister = _DMLLGPUCpp.SquareLossCpp_swigregister
SquareLossCpp_swigregister(SquareLossCpp)

class NeuralNetworkNodeGPUCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, NeuralNetworkNodeGPUCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, NeuralNetworkNodeGPUCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_NeuralNetworkNodeGPUCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_NeuralNetworkNodeGPUCpp
    __del__ = lambda self : None;
NeuralNetworkNodeGPUCpp_swigregister = _DMLLGPUCpp.NeuralNetworkNodeGPUCpp_swigregister
NeuralNetworkNodeGPUCpp_swigregister(NeuralNetworkNodeGPUCpp)

class ActivationFunctionGPUCpp(NeuralNetworkNodeGPUCpp):
    __swig_setmethods__ = {}
    for _s in [NeuralNetworkNodeGPUCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ActivationFunctionGPUCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NeuralNetworkNodeGPUCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, ActivationFunctionGPUCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_ActivationFunctionGPUCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_ActivationFunctionGPUCpp
    __del__ = lambda self : None;
ActivationFunctionGPUCpp_swigregister = _DMLLGPUCpp.ActivationFunctionGPUCpp_swigregister
ActivationFunctionGPUCpp_swigregister(ActivationFunctionGPUCpp)

class LogisticActivationFunctionGPUCpp(ActivationFunctionGPUCpp):
    __swig_setmethods__ = {}
    for _s in [ActivationFunctionGPUCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LogisticActivationFunctionGPUCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [ActivationFunctionGPUCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LogisticActivationFunctionGPUCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_LogisticActivationFunctionGPUCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_LogisticActivationFunctionGPUCpp
    __del__ = lambda self : None;
LogisticActivationFunctionGPUCpp_swigregister = _DMLLGPUCpp.LogisticActivationFunctionGPUCpp_swigregister
LogisticActivationFunctionGPUCpp_swigregister(LogisticActivationFunctionGPUCpp)

class LinearActivationFunctionGPUCpp(ActivationFunctionGPUCpp):
    __swig_setmethods__ = {}
    for _s in [ActivationFunctionGPUCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearActivationFunctionGPUCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [ActivationFunctionGPUCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LinearActivationFunctionGPUCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_LinearActivationFunctionGPUCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_LinearActivationFunctionGPUCpp
    __del__ = lambda self : None;
LinearActivationFunctionGPUCpp_swigregister = _DMLLGPUCpp.LinearActivationFunctionGPUCpp_swigregister
LinearActivationFunctionGPUCpp_swigregister(LinearActivationFunctionGPUCpp)

class NeuralNetworkGPUCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, NeuralNetworkGPUCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, NeuralNetworkGPUCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_NeuralNetworkGPUCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_NeuralNetworkGPUCpp
    __del__ = lambda self : None;
    def init_hidden_node(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_init_hidden_node(self, *args)
    def init_output_node(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_init_output_node(self, *args)
    def get_length_params(self): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_length_params(self)
    def get_params(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_params(self, *args)
    def get_input_nodes_fed_into_me_dense_length(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_input_nodes_fed_into_me_dense_length(self, *args)
    def get_input_nodes_fed_into_me_dense(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_input_nodes_fed_into_me_dense(self, *args)
    def get_input_nodes_fed_into_me_sparse_length(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_input_nodes_fed_into_me_sparse_length(self, *args)
    def get_input_nodes_fed_into_me_sparse(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_input_nodes_fed_into_me_sparse(self, *args)
    def get_hidden_nodes_fed_into_me_length(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_hidden_nodes_fed_into_me_length(self, *args)
    def get_hidden_nodes_fed_into_me(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_hidden_nodes_fed_into_me(self, *args)
    def finalise(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_finalise(self, *args)
    def load_dense_data(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_load_dense_data(self, *args)
    def load_dense_targets(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_load_dense_targets(self, *args)
    def load_sparse_data(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_load_sparse_data(self, *args)
    def load_sparse_targets(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_load_sparse_targets(self, *args)
    def fit(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_fit(self, *args)
    def transform(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_transform(self, *args)
    def get_sum_gradients_length(self): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_sum_gradients_length(self)
    def get_sum_gradients(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_sum_gradients(self, *args)
NeuralNetworkGPUCpp_swigregister = _DMLLGPUCpp.NeuralNetworkGPUCpp_swigregister
NeuralNetworkGPUCpp_swigregister(NeuralNetworkGPUCpp)

class OptimiserCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, OptimiserCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, OptimiserCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DMLLGPUCpp.new_OptimiserCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_OptimiserCpp
    __del__ = lambda self : None;
OptimiserCpp_swigregister = _DMLLGPUCpp.OptimiserCpp_swigregister
OptimiserCpp_swigregister(OptimiserCpp)

class SGDCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SGDCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, SGDCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_SGDCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_SGDCpp
    __del__ = lambda self : None;
SGDCpp_swigregister = _DMLLGPUCpp.SGDCpp_swigregister
SGDCpp_swigregister(SGDCpp)

# This file is compatible with both classic and new-style classes.


