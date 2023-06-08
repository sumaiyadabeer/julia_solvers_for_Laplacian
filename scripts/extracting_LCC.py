import argparse
from scipy import sparse
from scipy.sparse.csgraph import connected_components 
import numpy as np


# from itertools import count
# from re import L
# import igraph as ig
# from igraph import Graph

parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-f", "--filename", help = "Input Filename")
parser.add_argument("-i", "--index", help = "Index start from 0/1")

 
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
A = sparse.csr_matrix((vertex, vertex), dtype=np.int8)
A = A.todense()

print("Reading file and designing np matrix")

for line in Lines[1:]:
    l=line.split()
    if (int(l[0])) != (int(l[1])):
        A[int(l[0])-int(args.index),int(l[1])-int(args.index)] = 1
        A[int(l[1])-int(args.index),int(l[0])-int(args.index)] = 1
    else:
        print("removing self loop at ", int(l[0]))
# remove other than LCC

A = sparse.csr_matrix(A)
# print(A.data)
print("extracting LCC:")
n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)

print(n_components)
print(labels)
# # Assuming the LCC with 0 label
# file1 = open('wiki_bug_components', 'w')
# for i in labels:
#     file1.write(str(i)+"\t")   
# file1.close()

#modify A according to label
print("converting to dense array")
A = A.toarray()
print("converting to np array")
A = np.array(A)
print("original shape of matrix", (A.shape))
count = 0 # bcz size is changing with each iteration
for i, comp in enumerate(labels):
    if (comp):
        # print("delete ", i ,"th row and column")
        A = np.delete(A, i-count, 0) #deleting row
        A = np.delete(A, i-count, 1) #deleting column
        count = count + 1

print("Shape of LCC of matrix", (A.shape))


# Code for cross checking
# n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)
# for i in labels:
#     if i != 0:
#         print("it is not working", i)


n = A.shape[0]
e = np.count_nonzero(A)

# print(n)
# print(e)


#  to get lap
# LA = sparse.csgraph.laplacian(A)
# LA = sparse.csr_matrix(LA)
A = sparse.csr_matrix(A)

# write n, e graph

print("writing CSR graph")


file1 = open(name+'_csr', 'w')
file1.write(str(n) + "\t" +str(e) + "\n")  # writing line 1    

# write row_ptr \n
for i in A.indptr:    
    file1.write(str(i)+"\t")
file1.write("\n")

# write row_ptr \n
for i in A.indices:    
    file1.write(str(i)+"\t")
file1.write("\n")

# write row_ptr \n
for i in A.data:    
    file1.write(str(i)+"\t")
file1.write("\n")
   
file1.close()

print("writing IJV graph")
A = sparse.coo_matrix(A)

file1 = open(name+'_ijv', 'w')
file1.write(str(n) + "\t" +str(e) + "\n")  # writing line 1    


for i in range(A.row.size):  
    #adding "1" bcz of julia indexing  
    # print("writing", i)
    file1.write(str(A.row[i]+1) + "\t" + str(A.col[i]+1) + "\n")


file1.close()



print(" Conversion is done successfully with filename " + name+'_csr/_ijv ' )

