import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, parent_dir)
sys.path = list(dict.fromkeys(sys.path))
print(parent_dir)

# from modules.preproc import LogisticRegression as t
# print(t.test())