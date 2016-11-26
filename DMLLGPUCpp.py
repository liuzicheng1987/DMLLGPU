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

class RegulariserCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, RegulariserCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, RegulariserCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DMLLGPUCpp.new_RegulariserCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_RegulariserCpp
    __del__ = lambda self : None;
RegulariserCpp_swigregister = _DMLLGPUCpp.RegulariserCpp_swigregister
RegulariserCpp_swigregister(RegulariserCpp)

class L2RegulariserCpp(RegulariserCpp):
    __swig_setmethods__ = {}
    for _s in [RegulariserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, L2RegulariserCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [RegulariserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, L2RegulariserCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_L2RegulariserCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_L2RegulariserCpp
    __del__ = lambda self : None;
L2RegulariserCpp_swigregister = _DMLLGPUCpp.L2RegulariserCpp_swigregister
L2RegulariserCpp_swigregister(L2RegulariserCpp)

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

class SoftmaxActivationFunctionGPUCpp(ActivationFunctionGPUCpp):
    __swig_setmethods__ = {}
    for _s in [ActivationFunctionGPUCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SoftmaxActivationFunctionGPUCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [ActivationFunctionGPUCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, SoftmaxActivationFunctionGPUCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_SoftmaxActivationFunctionGPUCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_SoftmaxActivationFunctionGPUCpp
    __del__ = lambda self : None;
SoftmaxActivationFunctionGPUCpp_swigregister = _DMLLGPUCpp.SoftmaxActivationFunctionGPUCpp_swigregister
SoftmaxActivationFunctionGPUCpp_swigregister(SoftmaxActivationFunctionGPUCpp)

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
    def set_params(self, *args): return _DMLLGPUCpp.NeuralNetworkGPUCpp_set_params(self, *args)
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
    def get_sum_output_dim(self): return _DMLLGPUCpp.NeuralNetworkGPUCpp_get_sum_output_dim(self)
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

class AdaGradCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, AdaGradCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, AdaGradCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_AdaGradCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_AdaGradCpp
    __del__ = lambda self : None;
AdaGradCpp_swigregister = _DMLLGPUCpp.AdaGradCpp_swigregister
AdaGradCpp_swigregister(AdaGradCpp)

class LogicalGateCpp(NeuralNetworkNodeGPUCpp):
    __swig_setmethods__ = {}
    for _s in [NeuralNetworkNodeGPUCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LogicalGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NeuralNetworkNodeGPUCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LogicalGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_LogicalGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_LogicalGateCpp
    __del__ = lambda self : None;
LogicalGateCpp_swigregister = _DMLLGPUCpp.LogicalGateCpp_swigregister
LogicalGateCpp_swigregister(LogicalGateCpp)

class ANDGateCpp(LogicalGateCpp):
    __swig_setmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ANDGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, ANDGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_ANDGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_ANDGateCpp
    __del__ = lambda self : None;
ANDGateCpp_swigregister = _DMLLGPUCpp.ANDGateCpp_swigregister
ANDGateCpp_swigregister(ANDGateCpp)

class ORGateCpp(LogicalGateCpp):
    __swig_setmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ORGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, ORGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_ORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_ORGateCpp
    __del__ = lambda self : None;
ORGateCpp_swigregister = _DMLLGPUCpp.ORGateCpp_swigregister
ORGateCpp_swigregister(ORGateCpp)

class XORGateCpp(LogicalGateCpp):
    __swig_setmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, XORGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, XORGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_XORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_XORGateCpp
    __del__ = lambda self : None;
XORGateCpp_swigregister = _DMLLGPUCpp.XORGateCpp_swigregister
XORGateCpp_swigregister(XORGateCpp)

class XNORGateCpp(LogicalGateCpp):
    __swig_setmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, XNORGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, XNORGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_XNORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_XNORGateCpp
    __del__ = lambda self : None;
XNORGateCpp_swigregister = _DMLLGPUCpp.XNORGateCpp_swigregister
XNORGateCpp_swigregister(XNORGateCpp)

class NORGateCpp(LogicalGateCpp):
    __swig_setmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, NORGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, NORGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_NORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_NORGateCpp
    __del__ = lambda self : None;
NORGateCpp_swigregister = _DMLLGPUCpp.NORGateCpp_swigregister
NORGateCpp_swigregister(NORGateCpp)

class NANDGateCpp(LogicalGateCpp):
    __swig_setmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, NANDGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [LogicalGateCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, NANDGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLGPUCpp.new_NANDGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLGPUCpp.delete_NANDGateCpp
    __del__ = lambda self : None;
NANDGateCpp_swigregister = _DMLLGPUCpp.NANDGateCpp_swigregister
NANDGateCpp_swigregister(NANDGateCpp)

# This file is compatible with both classic and new-style classes.


