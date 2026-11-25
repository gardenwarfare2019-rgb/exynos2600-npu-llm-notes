# Section 4.5: compile fixed2.snc from Python in tuning_mode (output: WORKSPACE_DIR/Compile_tm).
# Usage: python3 compile_tuning.py WORKSPACE_DIR
import sys
from eais.workspace.workspace import get_workspace
W = sys.argv[1]
gw = get_workspace(W)
q = W + "/Converter_result/fixed2.snc"
r = gw.compiler.compile(input_model={"qsnc": q, "input_model": [q]}, output_dir=W + "/Compile_tm")
print("RESULT", r)
