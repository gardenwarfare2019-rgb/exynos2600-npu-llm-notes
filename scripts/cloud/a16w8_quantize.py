# Section 9, step 2: gamma fix inside the converted SNC, then the quantizer is run directly.
import snc_py_api, os, subprocess
W = "/home/exynos-lab/w_L2e10q/"
s = snc_py_api.Snc(); s.load_snc_file(W + "Converter_result/convert/w_L2e10q_new_snc.snc")
m = s.get_nn().get_models()[0]
ng = 0
for l in m.get_layers():
    if l.get_optype() == "RMSNormalization":
        k = l.get_tensors_map()["kernel"]
        g = k.get_numpy()
        k.set_numpy(g.reshape(1, int(g.size), 1, 1).copy())
        k.set_shape([1, int(g.size), 1, 1])
        ng += 1
print("gamma fixed", ng, flush=True)
s.save_snc_file(W + "Converter_result/convert/fixed_in.snc")
env = dict(os.environ)
env["PYTHONPATH"] = "/usr/local/lib/python3.10/dist-packages/eais/lib/ENNQuantizer:" + env.get("PYTHONPATH", "")
cmd = ('python3 /usr/local/lib/python3.10/dist-packages/eais/lib/ENNQuantizer/scripts/eais_conversion.py '
 '--snc_file "' + W + 'Converter_result/convert/fixed_in.snc" --out_fdr "' + W + 'Q_out" '
 '--preprocess DimensionalShift MergeConcat AdvProfiler --quantizer AsymQuant --calibration_method percentile '
 '--profiler_select AdvProfiler --postprocess OutportReshape --target_soc THETIS --bit_width_scale 12 '
 '--dataset ' + W + 'Config/DATA/database.txt --iteration 2 --dtype_edge INT16 --dtype_weight INT8 '
 '--input_dtypes FLOAT32 --output_dtypes FLOAT32')
r = subprocess.run(cmd, shell=True, env=env, cwd=W)
print("QRC", r.returncode, flush=True)
print(subprocess.run("find " + W + "Q_out -name '*.snc' -exec ls -la {} +", shell=True, capture_output=True, text=True).stdout)
