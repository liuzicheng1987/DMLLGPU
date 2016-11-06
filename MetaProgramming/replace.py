import os

os.chdir("/home/patrick/Dropbox/Programs/NeuralNetworkGPU/")

def replace(folder):
    flist = os.listdir(folder)
    
    for f in flist:
        print f
        if ".so" in f or ".o" in f or "Meta" in f or "pyc" in f or ".git" in f:
            continue
        if os.path.isdir(f):
            replace(folder + f + "/")
            continue
            
        text = open(folder + f, "rb").read()
        
        text = text.replace("NoWeightUpdates", "no_weight_updates")     
        text = text.replace("OutputPtr", "output_ptr")     
        text = text.replace("DeltaPtr", "delta_ptr")     
        
        open(folder + f, "wb").write(text)
            
replace("/home/patrick/Dropbox/Programs/NeuralNetworkGPU/")


