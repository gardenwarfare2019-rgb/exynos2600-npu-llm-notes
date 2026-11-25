# Section 4.4: RMSNorm gamma (kernel) to shape [1,C,1,1].
# Usage: python3 fix_gamma.py WORKSPACE_DIR   (reads fixed.snc, writes fixed2.snc)
import sys, snc_py_api
B = sys.argv[1] + "/Converter_result/"
s = snc_py_api.Snc(); s.load_snc_file(B + "fixed.snc")
m = s.get_nn().get_models()[0]
n = 0
for l in m.get_layers():
    if l.get_optype() == "RMSNormalization":
        k = l.get_tensors_map()["kernel"]
        a = k.get_numpy()
        k.set_numpy(a.reshape(1, int(a.size), 1, 1).copy())
        k.set_shape([1, int(a.size), 1, 1])
        n += 1
print("gamma fixed", n)
s.save_snc_file(B + "fixed2.snc")
