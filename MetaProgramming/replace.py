import os

os.chdir("/home/patrick/Dropbox/Programs/discovery/")

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
        
        text = text.replace("GPU", "")     
        text = text.replace("DMLL", "discovery")     
        
        open(folder + f, "wb").write(text)
            
replace("/home/patrick/Dropbox/Programs/discovery/")


