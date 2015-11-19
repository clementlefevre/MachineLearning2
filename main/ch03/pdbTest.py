# epdb1.py -- experiment with the Python debugger, pdb
a = "aaa"
b = "bbb"
c = "ccc"
import pdb
pdb.set_trace()
final = a + b + c
print final