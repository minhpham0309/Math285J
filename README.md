# Math285J

for windows:
copy file "FISTA_fast.pyd" to the folder Anaconda2\Lib\site-packages\sklearn\decomposition
copyt files "cdnmf_fast.pyd" and "nmf.py" and overide the "original" in the same folder
(make sure you make a copy of the original files before overiding, in case it crashes)


for linux, those files are "FISTA_fast.so" and "cdnmf_fast.so"

soure codes: FISTA_fast.pyx, cdnmf_fast.pyx and nmf.py

method: 
nmf = NMF(n_components=100, solver='cd',verbose=1,tol=.001,alpha=.1,l1_ratio=.2)
solver = 'cd': is the coordinate descent that I have modified
you can try other solver "ft" (for FISTA method)
