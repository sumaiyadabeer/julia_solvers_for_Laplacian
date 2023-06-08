import argparse
from itertools import count
from re import L
# import igraph as ig
# from igraph import Graph

parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-f", "--filename", help = "Input Filename")

 
# Read arguments from command line
args = parser.parse_args()

name=args.filename
file1 = open(name, 'r')
Lines = file1.readlines()
file1.close()

line = Lines[0]
l = line.split()

vertex = int(l[0])
edge = int(l[1])
# print(vertex)

IDs = []
for line in Lines[1:]:
    l=line.split()
    IDs.append(int(l[0]))
    IDs.append(int(l[1]))

ID_set = list(set(IDs))
mapping_IDs = list(range(1,len(ID_set)+1))
mapping = dict(zip(ID_set, mapping_IDs))
 
print(mapping)

file1 = open(name+'_processed', 'w')

file1.write(str(vertex)+ "\t" + str(edge)+ "\n")  # writing line 1    

for line in Lines[1:]:
    l=line.split()
    file1.write(str(mapping[int(l[0])])+ "\t" + str(mapping[int(l[1])])+ "\n")    
file1.close()


# print(Lines[1:])


