# Section 4.3: RMSNormalization axis from an array to a number.
# Usage: python3 fix_axis.py WORKSPACE_DIR   (reads the converted HW_*_new_snc_*.snc, writes fixed.snc)
import sys, glob, snc_py_api
B = sys.argv[1] + "/Converter_result/"
s = snc_py_api.Snc(); s.load_snc_file(glob.glob(B + "HW_*_new_snc_*.snc")[0])
m = s.get_nn().get_models()[0]
n = 0
for l in m.get_layers():
    if l.get_optype() == "RMSNormalization":
        a = l.get_params()["axis"]
        l.set_param("axis", int(a[0] if isinstance(a, (list, tuple)) else a))
        n += 1
print("axis fixed", n)
s.save_snc_file(B + "fixed.snc")
