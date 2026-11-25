# Same as compile_tuning.py but in the default compiler mode (output: WORKSPACE_DIR/Compile_out).
# Usage: python3 compile_default.py WORKSPACE_DIR
import sys
from eais.workspace.workspace import get_workspace
W = sys.argv[1]
gw = get_workspace(W)
q = W + "/Converter_result/fixed2.snc"
r = gw.compiler.compile(input_model={"qsnc": q, "input_model": [q]}, output_dir=W + "/Compile_out")
print("RESULT", r)
