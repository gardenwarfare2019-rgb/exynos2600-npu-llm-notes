# Section 9 (A16W8 workaround), step 1: init and conversion with activation_bitwidth 16.
# Example for chunk L2e10; the conversion is expected to fail at AdvProfiler.
import os, shutil, subprocess
H = os.path.expanduser("~") + "/"
W = H + "w_L2e10q"
def run(c):
    print(">>", c, flush=True)
    r = subprocess.run(c, shell=True)
    print("rc", r.returncode, flush=True)
os.makedirs(W, exist_ok=True)
shutil.copy(H + "q06_L2e10x0.onnx", W + "/w_L2e10q.onnx")
run("eais init --w " + W)
run("cd " + W + " && sed -i 's/^activation_bitwidth: 8/activation_bitwidth: 16/' Config/legacy_config.yaml && grep -nE '^(quantize_type|activation_bitwidth|bw_kernel|use_randomdb|model_type)' Config/legacy_config.yaml")
run("cd " + W + " && eais conversion --w " + W)
run("ls " + W + "/Converter_result " + W + "/Converter_result/convert " + W + "/Config/DATA")
print("DONE")
