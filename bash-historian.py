#!/usr/bin/env python3

import sys
import argparse

def all_same(l):
    assert len(l) > 0
    return l.count(l[0]) == len(l)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        metavar="in_history",
        nargs="+",
        type=argparse.FileType('r'),
        )
    globals().update(vars(parser.parse_args()))
    
    lines = dict()
    
    for fileno in range(len(files)):
        hist_file = files[fileno]
        lineno = 0
        for line in hist_file:
            if len(line) < 2:
                continue
            lineno += 1
            lines[line.strip()] = lineno + fileno / len(files)
    
    [print(k) for k, v in sorted(lines.items(), key=lambda line: line[1])]

if __name__=="__main__":
    main()

