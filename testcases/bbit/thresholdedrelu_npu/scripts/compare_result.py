import os
import sys
import numpy as np
import my_utils as my

is_save_log = False
# is_save_log = True

if not is_save_log:
    my.compare(1, 13, dtype='float32')
else:
    saved_stdout = sys.stdout
    print_log = open('compare_log.txt', 'w')
    sys.stdout = print_log
    my.compare(1, 13, dtype='float32')
    sys.stdout = saved_stdout
    print_log.close()

