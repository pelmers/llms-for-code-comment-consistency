#!/usr/bin/env python3

# Takes input on stdin, replaces all newlines with \n

import sys

if __name__ == "__main__":
    all_lines = []
    for line in sys.stdin:
        all_lines.append(line.strip().replace("\"", "\\\""))
    print("\\n".join(all_lines))