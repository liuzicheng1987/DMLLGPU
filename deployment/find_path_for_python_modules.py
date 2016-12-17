import os
import sys

for directory in sys.path[1:]:
    if os.path.isdir(directory):
        print directory
        break
