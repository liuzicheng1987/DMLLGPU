import os

os.chdir("/home/patrick/Dropbox/Programs/NeuralNetworkGPU/")

def replace(folder):
    flist = os.listdir(folder)
    
    for f in flist:
        print f
        if ".py" in f or ".so" in f or ".o" in f or "Meta" in f:
            continue
        if os.path.isdir(f):
            replace(folder + f + "/")
            continue
            
            text = open(f, "rb").read()
            
            if "unsigned" in text:
                print "Look into:" + f
            #text = text.replace("int", "std::uint32_t")
            
            open(f, "wb").write(text)
            
replace("/home/patrick/Dropbox/Programs/NeuralNetworkGPU/")


