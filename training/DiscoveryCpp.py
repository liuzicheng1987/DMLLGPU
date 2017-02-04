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
            fp, pathname, description = imp.find_module('_DiscoveryCpp', [dirname(__file__)])
        except ImportError:
            import _DiscoveryCpp
            return _DiscoveryCpp
        if fp is not None:
            try:
                _mod = imp.load_module('_DiscoveryCpp', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _DiscoveryCpp = swig_import_helper()
    del swig_import_helper
else:
    import _DiscoveryCpp
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
        this = _DiscoveryCpp.new_LossFunctionCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_LossFunctionCpp
    __del__ = lambda self : None;
LossFunctionCpp_swigregister = _DiscoveryCpp.LossFunctionCpp_swigregister
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
        this = _DiscoveryCpp.new_SquareLossCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_SquareLossCpp
    __del__ = lambda self : None;
SquareLossCpp_swigregister = _DiscoveryCpp.SquareLossCpp_swigregister
SquareLossCpp_swigregister(SquareLossCpp)

class RegulariserCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, RegulariserCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, RegulariserCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DiscoveryCpp.new_RegulariserCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_RegulariserCpp
    __del__ = lambda self : None;
RegulariserCpp_swigregister = _DiscoveryCpp.RegulariserCpp_swigregister
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
        this = _DiscoveryCpp.new_L2RegulariserCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_L2RegulariserCpp
    __del__ = lambda self : None;
L2RegulariserCpp_swigregister = _DiscoveryCpp.L2RegulariserCpp_swigregister
L2RegulariserCpp_swigregister(L2RegulariserCpp)

class NumericallyOptimisedAlgorithmCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, NumericallyOptimisedAlgorithmCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, NumericallyOptimisedAlgorithmCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DiscoveryCpp.new_NumericallyOptimisedAlgorithmCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_NumericallyOptimisedAlgorithmCpp
    __del__ = lambda self : None;
NumericallyOptimisedAlgorithmCpp_swigregister = _DiscoveryCpp.NumericallyOptimisedAlgorithmCpp_swigregister
NumericallyOptimisedAlgorithmCpp_swigregister(NumericallyOptimisedAlgorithmCpp)

class OptimiserCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, OptimiserCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, OptimiserCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DiscoveryCpp.new_OptimiserCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_OptimiserCpp
    __del__ = lambda self : None;
OptimiserCpp_swigregister = _DiscoveryCpp.OptimiserCpp_swigregister
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
        this = _DiscoveryCpp.new_SGDCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_SGDCpp
    __del__ = lambda self : None;
SGDCpp_swigregister = _DiscoveryCpp.SGDCpp_swigregister
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
        this = _DiscoveryCpp.new_AdaGradCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_AdaGradCpp
    __del__ = lambda self : None;
AdaGradCpp_swigregister = _DiscoveryCpp.AdaGradCpp_swigregister
AdaGradCpp_swigregister(AdaGradCpp)

class RMSPropCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, RMSPropCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, RMSPropCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_RMSPropCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_RMSPropCpp
    __del__ = lambda self : None;
RMSPropCpp_swigregister = _DiscoveryCpp.RMSPropCpp_swigregister
RMSPropCpp_swigregister(RMSPropCpp)

class AdamCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, AdamCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, AdamCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_AdamCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_AdamCpp
    __del__ = lambda self : None;
AdamCpp_swigregister = _DiscoveryCpp.AdamCpp_swigregister
AdamCpp_swigregister(AdamCpp)

class NadamCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, NadamCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, NadamCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_NadamCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_NadamCpp
    __del__ = lambda self : None;
NadamCpp_swigregister = _DiscoveryCpp.NadamCpp_swigregister
NadamCpp_swigregister(NadamCpp)

class NeuralNetworkNodeCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, NeuralNetworkNodeCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, NeuralNetworkNodeCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_NeuralNetworkNodeCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_NeuralNetworkNodeCpp
    __del__ = lambda self : None;
NeuralNetworkNodeCpp_swigregister = _DiscoveryCpp.NeuralNetworkNodeCpp_swigregister
NeuralNetworkNodeCpp_swigregister(NeuralNetworkNodeCpp)

class ActivationFunctionCpp(NeuralNetworkNodeCpp):
    __swig_setmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ActivationFunctionCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, ActivationFunctionCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_ActivationFunctionCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_ActivationFunctionCpp
    __del__ = lambda self : None;
ActivationFunctionCpp_swigregister = _DiscoveryCpp.ActivationFunctionCpp_swigregister
ActivationFunctionCpp_swigregister(ActivationFunctionCpp)

class LogisticActivationFunctionCpp(ActivationFunctionCpp):
    __swig_setmethods__ = {}
    for _s in [ActivationFunctionCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LogisticActivationFunctionCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [ActivationFunctionCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LogisticActivationFunctionCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_LogisticActivationFunctionCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_LogisticActivationFunctionCpp
    __del__ = lambda self : None;
LogisticActivationFunctionCpp_swigregister = _DiscoveryCpp.LogisticActivationFunctionCpp_swigregister
LogisticActivationFunctionCpp_swigregister(LogisticActivationFunctionCpp)

class LinearActivationFunctionCpp(ActivationFunctionCpp):
    __swig_setmethods__ = {}
    for _s in [ActivationFunctionCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearActivationFunctionCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [ActivationFunctionCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LinearActivationFunctionCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_LinearActivationFunctionCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_LinearActivationFunctionCpp
    __del__ = lambda self : None;
LinearActivationFunctionCpp_swigregister = _DiscoveryCpp.LinearActivationFunctionCpp_swigregister
LinearActivationFunctionCpp_swigregister(LinearActivationFunctionCpp)

class SoftmaxActivationFunctionCpp(ActivationFunctionCpp):
    __swig_setmethods__ = {}
    for _s in [ActivationFunctionCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SoftmaxActivationFunctionCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [ActivationFunctionCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, SoftmaxActivationFunctionCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_SoftmaxActivationFunctionCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_SoftmaxActivationFunctionCpp
    __del__ = lambda self : None;
SoftmaxActivationFunctionCpp_swigregister = _DiscoveryCpp.SoftmaxActivationFunctionCpp_swigregister
SoftmaxActivationFunctionCpp_swigregister(SoftmaxActivationFunctionCpp)

class NeuralNetworkCpp(NumericallyOptimisedAlgorithmCpp):
    __swig_setmethods__ = {}
    for _s in [NumericallyOptimisedAlgorithmCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, NeuralNetworkCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NumericallyOptimisedAlgorithmCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, NeuralNetworkCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_NeuralNetworkCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_NeuralNetworkCpp
    __del__ = lambda self : None;
    def init_hidden_node(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_init_hidden_node(self, *args)
    def init_output_node(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_init_output_node(self, *args)
    def get_length_params(self): return _DiscoveryCpp.NeuralNetworkCpp_get_length_params(self)
    def get_params(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_params(self, *args)
    def set_params(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_set_params(self, *args)
    def get_input_nodes_fed_into_me_dense_length(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_input_nodes_fed_into_me_dense_length(self, *args)
    def get_input_nodes_fed_into_me_dense(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_input_nodes_fed_into_me_dense(self, *args)
    def get_input_nodes_fed_into_me_sparse_length(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_input_nodes_fed_into_me_sparse_length(self, *args)
    def get_input_nodes_fed_into_me_sparse(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_input_nodes_fed_into_me_sparse(self, *args)
    def get_hidden_nodes_fed_into_me_length(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_hidden_nodes_fed_into_me_length(self, *args)
    def get_hidden_nodes_fed_into_me(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_hidden_nodes_fed_into_me(self, *args)
    def finalise(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_finalise(self, *args)
    def load_dense_data(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_load_dense_data(self, *args)
    def load_dense_targets(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_load_dense_targets(self, *args)
    def load_sparse_data(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_load_sparse_data(self, *args)
    def load_sparse_targets(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_load_sparse_targets(self, *args)
    def fit(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_fit(self, *args)
    def transform(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_transform(self, *args)
    def get_sum_gradients_length(self): return _DiscoveryCpp.NeuralNetworkCpp_get_sum_gradients_length(self)
    def get_sum_gradients(self, *args): return _DiscoveryCpp.NeuralNetworkCpp_get_sum_gradients(self, *args)
    def get_sum_output_dim(self): return _DiscoveryCpp.NeuralNetworkCpp_get_sum_output_dim(self)
NeuralNetworkCpp_swigregister = _DiscoveryCpp.NeuralNetworkCpp_swigregister
NeuralNetworkCpp_swigregister(NeuralNetworkCpp)

class RelationalNetworkCpp(NumericallyOptimisedAlgorithmCpp):
    __swig_setmethods__ = {}
    for _s in [NumericallyOptimisedAlgorithmCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, RelationalNetworkCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NumericallyOptimisedAlgorithmCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, RelationalNetworkCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DiscoveryCpp.new_RelationalNetworkCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_RelationalNetworkCpp
    __del__ = lambda self : None;
    def add_input_network(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_add_input_network(self, *args)
    def set_output_network(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_set_output_network(self, *args)
    def finalise(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_finalise(self, *args)
    def load_dense_data(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_load_dense_data(self, *args)
    def load_dense_targets(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_load_dense_targets(self, *args)
    def load_time_stamps_output(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_load_time_stamps_output(self, *args)
    def load_time_stamps_input(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_load_time_stamps_input(self, *args)
    def add_join_keys_left(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_add_join_keys_left(self, *args)
    def clean_up(self): return _DiscoveryCpp.RelationalNetworkCpp_clean_up(self)
    def fit(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_fit(self, *args)
    def transform(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_transform(self, *args)
    def get_sum_output_dim(self): return _DiscoveryCpp.RelationalNetworkCpp_get_sum_output_dim(self)
    def get_sum_gradients_length(self): return _DiscoveryCpp.RelationalNetworkCpp_get_sum_gradients_length(self)
    def get_sum_gradients(self, *args): return _DiscoveryCpp.RelationalNetworkCpp_get_sum_gradients(self, *args)
RelationalNetworkCpp_swigregister = _DiscoveryCpp.RelationalNetworkCpp_swigregister
RelationalNetworkCpp_swigregister(RelationalNetworkCpp)

class LogicalGateCpp(NeuralNetworkNodeCpp):
    __swig_setmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LogicalGateCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LogicalGateCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_LogicalGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_LogicalGateCpp
    __del__ = lambda self : None;
LogicalGateCpp_swigregister = _DiscoveryCpp.LogicalGateCpp_swigregister
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
        this = _DiscoveryCpp.new_ANDGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_ANDGateCpp
    __del__ = lambda self : None;
ANDGateCpp_swigregister = _DiscoveryCpp.ANDGateCpp_swigregister
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
        this = _DiscoveryCpp.new_ORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_ORGateCpp
    __del__ = lambda self : None;
ORGateCpp_swigregister = _DiscoveryCpp.ORGateCpp_swigregister
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
        this = _DiscoveryCpp.new_XORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_XORGateCpp
    __del__ = lambda self : None;
XORGateCpp_swigregister = _DiscoveryCpp.XORGateCpp_swigregister
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
        this = _DiscoveryCpp.new_XNORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_XNORGateCpp
    __del__ = lambda self : None;
XNORGateCpp_swigregister = _DiscoveryCpp.XNORGateCpp_swigregister
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
        this = _DiscoveryCpp.new_NORGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_NORGateCpp
    __del__ = lambda self : None;
NORGateCpp_swigregister = _DiscoveryCpp.NORGateCpp_swigregister
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
        this = _DiscoveryCpp.new_NANDGateCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_NANDGateCpp
    __del__ = lambda self : None;
NANDGateCpp_swigregister = _DiscoveryCpp.NANDGateCpp_swigregister
NANDGateCpp_swigregister(NANDGateCpp)

class DropoutCpp(NeuralNetworkNodeCpp):
    __swig_setmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, DropoutCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, DropoutCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_DropoutCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_DropoutCpp
    __del__ = lambda self : None;
DropoutCpp_swigregister = _DiscoveryCpp.DropoutCpp_swigregister
DropoutCpp_swigregister(DropoutCpp)

class NodeSamplerCpp(NeuralNetworkNodeCpp):
    __swig_setmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, NodeSamplerCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, NodeSamplerCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_NodeSamplerCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_NodeSamplerCpp
    __del__ = lambda self : None;
NodeSamplerCpp_swigregister = _DiscoveryCpp.NodeSamplerCpp_swigregister
NodeSamplerCpp_swigregister(NodeSamplerCpp)

class AggregationCpp(NeuralNetworkNodeCpp):
    __swig_setmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, AggregationCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NeuralNetworkNodeCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, AggregationCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_AggregationCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_AggregationCpp
    __del__ = lambda self : None;
AggregationCpp_swigregister = _DiscoveryCpp.AggregationCpp_swigregister
AggregationCpp_swigregister(AggregationCpp)

class SumCpp(AggregationCpp):
    __swig_setmethods__ = {}
    for _s in [AggregationCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SumCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [AggregationCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, SumCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_SumCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_SumCpp
    __del__ = lambda self : None;
SumCpp_swigregister = _DiscoveryCpp.SumCpp_swigregister
SumCpp_swigregister(SumCpp)

class AvgCpp(AggregationCpp):
    __swig_setmethods__ = {}
    for _s in [AggregationCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, AvgCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [AggregationCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, AvgCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_AvgCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_AvgCpp
    __del__ = lambda self : None;
AvgCpp_swigregister = _DiscoveryCpp.AvgCpp_swigregister
AvgCpp_swigregister(AvgCpp)

class CountCpp(AggregationCpp):
    __swig_setmethods__ = {}
    for _s in [AggregationCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, CountCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [AggregationCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, CountCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_CountCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_CountCpp
    __del__ = lambda self : None;
CountCpp_swigregister = _DiscoveryCpp.CountCpp_swigregister
CountCpp_swigregister(CountCpp)

class FirstCpp(AggregationCpp):
    __swig_setmethods__ = {}
    for _s in [AggregationCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, FirstCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [AggregationCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, FirstCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_FirstCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_FirstCpp
    __del__ = lambda self : None;
FirstCpp_swigregister = _DiscoveryCpp.FirstCpp_swigregister
FirstCpp_swigregister(FirstCpp)

class LastCpp(AggregationCpp):
    __swig_setmethods__ = {}
    for _s in [AggregationCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LastCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [AggregationCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LastCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DiscoveryCpp.new_LastCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DiscoveryCpp.delete_LastCpp
    __del__ = lambda self : None;
LastCpp_swigregister = _DiscoveryCpp.LastCpp_swigregister
LastCpp_swigregister(LastCpp)

# This file is compatible with both classic and new-style classes.


