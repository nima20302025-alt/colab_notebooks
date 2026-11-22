import lightgbm
import lightgbm.libpath as lp
import os
print('lightgbm.__file__=', lightgbm.__file__)
print('lib paths =', lp._find_lib_path())
for p in lp._find_lib_path():
    if os.path.exists(p):
        print('exists:', p)
    else:
        print('missing:', p)
