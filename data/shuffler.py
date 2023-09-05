import os, sys, json
import random

fns = sys.argv[1:]

for fn in fns:
    with open(fn) as f:
        data = json.load(f)
        # if the data is a list
        if isinstance(data, list):
            random.shuffle(data)
        else:
            continue

    with open(fn, 'w') as f:
        json.dump(data, f, indent=2)

print('done')
