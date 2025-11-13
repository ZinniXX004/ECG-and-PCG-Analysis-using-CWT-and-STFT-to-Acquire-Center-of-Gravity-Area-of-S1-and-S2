"""
Small test script to check WFDB record loading and channels.
Usage: python testdatasetforecgandpcg.py a0007
"""

import sys
import wfdb

def main():
    if len(sys.argv) < 2:
        print("Usage: python testdatasetforecgandpcg.py <recordname>")
        return
    recname = sys.argv[1]
    rec = wfdb.rdrecord(recname)
    print("fs =", rec.fs)
    print("sig_name =", rec.sig_name)
    print("units =", rec.units)
    print("p_signal shape:", rec.p_signal.shape)
    # print a small sample of the first channel values
    print("first 10 samples (channel0):", rec.p_signal[:10, 0])

if __name__ == "__main__":
    main()
