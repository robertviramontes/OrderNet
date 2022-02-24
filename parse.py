# for parsing the logs for wirelength

import argparse
import re
from typing import List #, TypedDict
import pandas as pd

# class IterationData(TypedDict):
#     itr: int # which iteration the data is from
#     wl: int # wirelength at the end of that iteration
#     drc: int # drc at the end of that iteration

parser = argparse.ArgumentParser(description='Process my log files.')
parser.add_argument('filename', help="path to the log file to parse")
args = parser.parse_args()

with open(args.filename) as f:
    file_contents = f.readlines()

optimization_pattern = re.compile(r"Start (\d+).* optimization")
wirelength_pattern = re.compile(r"Total wire length = (\d+) um")
drc_pattern = re.compile(r"\[INFO DRT-0199\]   Number of violations = (\d+).")

log_data = [] # : List[IterationData] = []

for line in file_contents:
    if "[INFO DRT-0198] Complete detail routing." in  line:
        break        
    m = optimization_pattern.search(line)
    if m is not None:
        itr = int(m.group(1))
        continue

    d = drc_pattern.search(line)
    if d is not None:
        drc = int(d.group(1))
        continue
    
    n = wirelength_pattern.search(line)
    if n is not None:
        wl = int(n.group(1))
        log_data.append(
            {'itr': itr, 'wl': wl, 'drc': drc}
            )
        continue

itrs: List[int] = []
drcs: List[int] = []
wls: List[int] = []
for d in log_data:
    itrs.append(d['itr'])
    wls.append(d['wl'])
    drcs.append(d['drc'])

df = pd.DataFrame(data = {'Iteration': itrs, 'DRC Violations': drcs, 'Wirelength': wls})
print(df)
