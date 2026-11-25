# Whole FP16 pipeline for one ONNX chunk (NPU_GUIDE section 4).
# Usage (Exynos Lab notebook, run from Popen): python3 ~/build_chunk.py NAME FILE.onnx
# FILE.onnx is relative to the home folder; result: ~/out/NAME_tm.nnc.
# Return code 1 of "eais conversion" is expected (section 4.2).
import sys, os, subprocess, glob, shutil, re
H = os.path.expanduser('~') + '/'
N = sys.argv[1]
SRC = H + sys.argv[2]
W = H + 'w_' + N
assert os.path.exists(SRC), 'onnx not found'
assert not os.path.exists(W), 'folder already exists: ' + W
def run(c):
    print('>>', c, flush=True)
    r = subprocess.run(c, shell=True)
    print('rc', r.returncode, flush=True)
def edit(p, pat, rep):
    s = open(p).read()
    s2, n = re.subn(pat, rep, s, flags=re.M)
    open(p, 'w').write(s2)
    print('EDIT', os.path.basename(p), n, 'replacements', flush=True)
os.makedirs(W)
shutil.copy(SRC, W + '/w_' + N + '.onnx')
run('eais init --w ' + W)
for p in (W + '/Config/legacy_config.yaml', W + '/w_' + N + '.yaml'):
    edit(p, r'^(\s*(?:input|output)_dtype:\s*)float32', r'\1float16')
edit(W + '/Config/legacy_config.yaml', r'^quantize_type: asymm', 'quantize_type: fp16')
edit(W + '/w_' + N + '.yaml', r'^(\s*mode:\s*)\S+(\s+#\[default, multicore, tuning_mode)', r'\1tuning_mode\2')
run('cd ' + W + ' && grep -rn "_dtype:\\|quantize_type:\\|^  mode:" Config/legacy_config.yaml w_' + N + '.yaml | cut -c1-90')
run('cd ' + W + ' && eais conversion --w ' + W)
run('python3 ' + H + 'fix_axis.py ' + W)
run('python3 ' + H + 'fix_gamma.py ' + W)
os.makedirs(W + '/Compile_tm', exist_ok=True)
run('python3 ' + H + 'compile_tuning.py ' + W)
f = glob.glob(W + '/Compile_tm/**/*.nnc', recursive=True)
print('NNC', f)
os.makedirs(H + 'out', exist_ok=True)
for x in f:
    shutil.copy(x, H + 'out/' + N + '_tm.nnc')
print('DONE', N)
