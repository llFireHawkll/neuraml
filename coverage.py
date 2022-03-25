import os
import re
import sys


def coverage_check():
    source_path = "/tests"
    folder = os.getcwd() + source_path
    coverage = os.popen(f"coverage run -m pytest && " f"coverage report -m").read()
    print(coverage)
    coverages = re.search("test_[a-zA-Z0-9_]*.py.*", coverage)
    print(coverages)
    if coverages:
        coverage_group = coverages.group().replace(", ", ",")
        fname, stmts, miss, cover, missing = re.findall(r"\S+", coverage_group)
        stmts = int(stmts)
        miss = int(miss)
        cover = cover.split("%")[0]
        if int(cover) < 90:
            print(f"Increase coverage for {fname}")
            print(f"Missing lines are {missing}")
            sys.exit(os.EX_CONFIG)
    else:
        print(f"Test Cases Failed for {folder}")
        sys.exit(os.EX_CONFIG)

    sys.exit(os.EX_OK)


if __name__ == "__main__":
    coverage_check()
