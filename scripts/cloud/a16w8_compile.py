# Section 9, step 3: axis to a number in HW_fixed_in.snc, then compile.
# Usage: python3 a16w8_compile.py OUTPUT_FOLDER_NAME
import snc_py_api, os, sys
from eais.workspace.workspace import get_workspace
W = "/home/exynos-lab/w_L2e10q"
Q = W + "/Q_out/"
out = sys.argv[1]
if not os.path.exists(Q + "HW_axis.snc"):
    s = snc_py_api.Snc(); s.load_snc_file(Q + "HW_fixed_in.snc")
    m = s.get_nn().get_models()[0]
    n = 0
    for l in m.get_layers():
        if l.get_optype() == "RMSNormalization":
            a = l.get_params()["axis"]
            l.set_param("axis", int(a[0] if isinstance(a, (list, tuple)) else a))
            n += 1
    print("axis fixed", n, flush=True)
    s.save_snc_file(Q + "HW_axis.snc")
os.makedirs(W + "/" + out, exist_ok=True)
gw = get_workspace(W)
q = Q + "HW_axis.snc"
r = gw.compiler.compile(input_model={"qsnc": q, "input_model": [q]}, output_dir=W + "/" + out)
print("RESULT", r, flush=True)
