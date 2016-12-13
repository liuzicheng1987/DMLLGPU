import os

os.chdir("/home/patrick/Dropbox/Programs/DMLL/NeuralNetworkGPU/")
files = os.listdir("/home/patrick/Dropbox/Programs/DMLL/NeuralNetworkGPU/")

for f in files:
    content = open(f, "rb").read()
    open(f, "wb").write(content.replace("float", "float"))
