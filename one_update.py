import os
import shutil
from time import sleep
from linecache import getline


class File:

    def __init__(self, file_name):
        self.delete_me = False
        self.file_name = file_name
        file_name, suffix = file_name.split(".")
        if(suffix != "out"): self.delete_me = True ; return
        self.slurm_name = file_name.split("_")[0]

        self.name_for_user = "Starting..."
        lines = open(self.file_name).readlines()
        for i, line in enumerate(lines):
            if(line[:5] == "name:" and lines[i+1][0] != "\t"):
                self.name_for_user = lines[i+1][:-1]
                break

        self.last_line = "Starting..."
        if(len(lines) != 0):
            last = -1
            while(lines[last] == "\n"): last -= 1
            self.last_line = lines[last]

    def done(self):
        if(self.last_line.endswith("Done!\n")): return(True)
        return(False)

    def __lt__(self, other): return(self.slurm_name < other.slurm_name)



file_names = [f for f in os.listdir() if f[:5]=="slurm"]

files = []
for name in file_names:
    file = File(name)
    if(file.delete_me): pass
    else:               files.append(file)
files.sort()

ongoing = []              ; finished = []

for file in files:
    done = file.done()
    if(done): finished.append(file)
    else:     ongoing.append(file)


print("\n\n")

print("Ongoing:")
if(len(ongoing) == 0): print("None.")
for file in ongoing:
    last_line = file.last_line if file.last_line[-1] != "\n" else file.last_line[:-1]
    print("{} ({}): \t{}".format(file.name_for_user, file.file_name, last_line))

print("\n\n\nFinished:")
if(len(finished) == 0): print("None.")
else:
    for i, file in enumerate(finished):
        last_line = file.last_line if file.last_line[-1] != "\n" else file.last_line[:-1]
        print("{} ({}): \t{}".format(file.name_for_user, file.file_name, last_line))

