
# GP-Lsolve

A brief description of execution of GP-Lsolve and comparision baselines

## Baselines for comparision

| List of baselines |
| ----------------- | 
| Cholesky-23       |
| CMG               | 
| Kyng & Sachdeva   | 
| PyAMG             | 


## Dataset for comparision

| List of baselines |
| ----------------- | 
| WikiBug           | 
| - | 
| - | 
| - | 


## Paper Link

Working implementation of GP-Lsolve 

## Data_preprocessing



* Select a dataset 
    
* Argue why running Lap solver is useful on it 

* Prepare the dataset 

    * Check for connectedness (ectract_LCC.py)

    * Check for multi-edges (deceiving degrees) 

    * Convert IDs to nodes in graph (id_convert.py)

* Prepare the inputs (G, and b)

    * Save CSR (on obelix) and IJV (on OptiPlex)

    * Select the sink and sources  (notebook design_b)
        * set the required number+1  of dense source nodes
        * select the highest degree node as sink



## Execution

### Runnning the project

Runnning GP-Lsolve 
```bash
cp data/wiki_bug/wikipedia_link_bug_csr_backup data/wiki_bug/wikipedia_link_bug_csr
cat data/wiki_bug/bug_25_b.txt >> data/wiki_bug/wikipedia_link_bug_csr
./main /home/sumaiya/Cuda_solver_optimization1/data/wiki_bug/wikipedia_link_bug_csr data/wiki_bug/wikipedia_link_bug_b_25_beta_1.out
```
Running pyamg 

Calculation of Lsolve residual
```bash
python3 pyamg_demo.py -csr /home/sumaiya/Cuda_solver_optimization1/data/wiki_bug/wikipedia_link_bug_csr -lapFile /home/sumaiya/Cuda_solver_optimization1/data/wiki_bug/wikipedia_link_bug_b_25.out -pyamFile /home/sumaiya/Cuda_solver_optimization1/data/wiki_bug/pyamg_b_25.out 
```

Running Julia solvers for CMG, KYNG and Chol23
```bash
sumaiya@sumaiya-OptiPlex-7050:~/Desktop/Datasets/julia_solvers$ julia solvers.jl wiki_bug/wikipedia_link_bug_ijv wiki_bug/bug_12_b.txt wiki_bug/results/b12
```
Renaming files (will remove later) "b50_Lsolve"

plotting using jupyter notebook (plotting.ipynb)

## Authors

- [@sumaiya](www.cse.iitd.ac.in/~sumaiya/)

