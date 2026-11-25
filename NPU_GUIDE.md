# NPU_GUIDE.md: building and running your own models on the Exynos 2600 NPU

A recipe based on LLAMA-ALT-01_LOG.md (numbers, dead ends and the result are there). Labels: see README. References use the §N.M format. The cloud scripts are included (scripts/cloud). The phone-side scripts and the test app are not; section 0 describes them by role.

## 0. Components (described by role)
Chunk names: `L{a}e{b}` means layers a to b-1; suffix `h` is the float16 interface, `tm` is tuning_mode, `q` is A16W8.

Cloud side (Exynos Lab notebook, Python 3.10; the scripts are in scripts/cloud):
- `build_chunk`: one command that runs the whole FP16 pipeline for one ONNX chunk (§4). It stops if the ONNX is missing or the workspace folder already exists.
- `fix_axis`: edits the converted graph (SNC) through `snc_py_api`: RMSNormalization axis from an array to a number (§4.3).
- `fix_gamma`: the same API: RMSNorm gamma from [1,1,1,C] to [1,C,1,1] (§4.4).
- `compile_tuning` and `compile_default`: call the compiler from Python (§4.5).
- `a16w8_init`, `a16w8_quantize`, `a16w8_compile`: the A16W8 workaround (§9).

Phone side (Termux with Python, clang, adb):
- `make_chunk_onnx`: writes ONNX by hand through protobuf for a layer range, an attention variant and a cache length (§5).
- `check_onnx_equivalence`: compares a generated ONNX with the reference.
- `export_head_weights`: prepares the int8 lm_head, the bf16 embedding and the final norm weights (§5).
- `device_accuracy_checks`: cosine against ONNX for a single chunk, the whole chain, several layers, and the FP16 chunks (§6).
- `device_bench` and `device_ab_bench`: speed of a chain or of one chunk (§7).
- `energy_charge_counter`: energy measurement (§7).
- `generation_runner`: writes the loop config, runs the C generation loop and compares it with the reference (§7).
- `preflight_check`: verifies the chunk files and the installed APK before any measurement.

Android test app (C with the NDK, no UI), three builds: single-exec (runs one chunk file), bench, generation-loop (three iterations, the last one is final). The build recipe is in §7.

## 1. Where things are
- Portal per.semiconductor.samsung.com, Development, Exynos Lab. Notebook Session: Framework AI Studio SDK, 8 cores, 32 GB RAM, 100 GB disk; after Launch Now a JupyterLab opens at ko.ai-studio-farm.com. FACT.
- No limits on runs or hours were found in the documentation. THEORY: files vanish when the session stops, so download results immediately.
- Do not open one notebook on the phone and on a PC at the same time (a File Changed dialog appears, choose Revert). A PC is more convenient.
- Moving files: phone, cable, PC, the JupyterLab file panel (up arrow); back through Download in Jupyter one file at a time; after downloading compare size and md5.

## 2. Tools in the cloud (FACT)
- eais 2026.4.21 (package 8.16.21.25), Python 3.10, onnx 1.17.0, onnxruntime 1.22.0, torch 1.13.1+cpu, snc_py_api, nncontainer-gen (the compiler). Commands: init, conversion, optimize, quantize, compile, profiling, runtime, simulation, verify, nnc-reader; group system: config, config-set, info. Minimal keys: --w FOLDER, everything else lives in configs.
- Config templates: /usr/local/lib/python3.10/dist-packages/eais/asset/config/.
- The Samsung web service (package 8.15.5.20) only suits A8W8 with random calibration, FP16 cannot be set in its form; use the cloud for your own models.

## 3. Workspace and settings (FACT)
- Workspace folder ~/w_NAME (or ~/ws) containing w_NAME.onnx (the file name equals the folder name, otherwise eais init complains). w_NAME.yaml and Config/ are created.
- w_NAME.yaml: device Gen-8 (for Exynos 2600), general mode legacy (options legacy, pro, gen-ai-eht, gen-ai-elt, gen-ai; gen-ai is unavailable in the cloud), compiler mode (default, multicore, tuning_mode), input_dtype and output_dtype float32 or float16, model_type CV.
- Config/legacy_config.yaml: quantize_type (symm, asymm, fp16, qat, bf16, mxint), activation_bitwidth (8, 16), bw_kernel (4, 8, 16), calibration_method, use_randomdb (true: random calibration), userdb, model_type, input_dtype, output_dtype.
- Config/DATA/dbgen_spec.yaml: your own calibration data (per input: path, dtype, ext, raw_shape). Config/quantize_tuning_mode_config.yaml: skip_quantization, calibration_data_path, mpq_name_dict, advanced_quantization_algorithm (smoothquant, adaround). For tuning_mode: Config/compiler_tuning_mode_config.yaml (cast_inout, quant_type, dequant_type hw, multicore true).
- Jupyter rules: commands ending with & are rejected; run long jobs from Python: subprocess.Popen(command, shell=True, start_new_session=True) with the output going to a log; watch with tail -n 40 LOG | cut -c1-200 and pgrep -fa eais. Create files with %%writefile as the first line of a cell. In a cell with an exclamation mark a dollar sign substitutes a Python variable: write paths explicitly. Run the chunks one after another, not in parallel.

## 4. Recipe: build an FP16 chunk (the working path, FACT)
One command (scripts in the cloud home): python3 ~/build_chunk.py NAME FILE.onnx (FILE.onnx is relative to the home folder), started from Popen, log ~/log_NAME.txt. Result: ~/out/NAME_tm.nnc. About 45 s per two-layer chunk. The script stops by itself if the onnx is missing or the folder w_NAME already exists.
What it does:
1. §4.1 eais init, then replace float32 with float16 in input_dtype and output_dtype (legacy_config.yaml and w_NAME.yaml), set quantize_type fp16 and compiler mode tuning_mode.
2. §4.2 eais conversion. Return code 1 is expected: the ICMI simulator does not know RMSNormalization (error generate_tv: H and W dim for gamma or beta should be 1); with float16 there is also an onnxruntime error on OutportReshape. The SNC files are written anyway (Converter_result/HW_*_new_snc_*.snc). check_output_snr false does not help (test vector generation is unconditional).
3. §4.3 fix_axis: the axis of every RMSNormalization from the array [1] to a number (compiler error: Get axis failed, type must be number, but is array). Result: fixed.snc.
4. §4.4 fix_gamma: gamma (kernel) from [1,1,1,C] to [1,C,1,1] (C is 1024 for the layer norms and 128 for q_norm and k_norm; the values are the same). Without it: CDI getHandler failed, Type RmsNorm. Result: fixed2.snc. Both edits go through snc_py_api: get_layers, get_optype, get_params, set_param, get_tensors_map, get_numpy, set_numpy, set_shape, save_snc_file.
5. §4.5 compile_tuning: gw = get_workspace(path); gw.compiler.compile(input_model={qsnc: q, input_model: [q]}, output_dir=FOLDER/Compile_tm). input_model only as a dictionary (a list is expanded into positional arguments and breaks the call). This detour is needed because eais compile, after the §4.2 failure, says "model needs to be converted".
6. Check: the log shows return codes by stage 0 0 1 0 0 0; in Compile_tm/NNCInfo.csv both input and output are FLOAT16; input size is 1408 + 260096 x layers, output size 1024 x (1 + 2 x layers); for chunk 0-1 the input is 521600 and the output 5120 (examples/NNCInfo_*.csv are the real files for three chunks).
Default mode instead of tuning_mode: keep compiler mode default in w_NAME.yaml and use compile_default (output in Compile_out). tuning_mode gives about minus 9 percent of time.

## 5. The model: ONNX for the NPU (FACT)
- torch.onnx.export does not work in Termux: the ONNX is written by hand through protobuf, only standard operations, opset 17, weights fp16 + Cast, RMSNorm as Div(1, Sqrt) + Mul x + Mul weight, norms in fp32.
- Chunk interface: one input inp [1,1,N] fp32 and one output out (the web service failed with several inputs and outputs; a multi-input variant was not tried in the cloud). Input: x 1024, cos 128, sin 128, mask 128, then K and V of every layer, 130048 each (8 heads x 127 positions x 128); input size 1408 + 260096 x layers. Output: hout 1024, then nk and nv of every layer, 1024 each (1024 x (1 + 2 x layers)). Chunks are cut by ranges: L0e2 (layers 0-1), L2e16 (2-15), L16e28 (16-27) and others.
- Generators on the phone exist as several variants: a range generator (arguments A B, layers A..B-1), a fixed-range generator for layers 2-6 (the argument 0 is mandatory), a generator for layers 16-28, a variant generator (arguments PART VAR CTX: attention variants O, A, B, C, E; CTX is the cache length 32, 64 or 128) and a helper that makes copies by range. The self-check inside the fixed-range generator is dead (an early sys.exit(0)): check equivalence with check_onnx_equivalence.
- Context: the shorter the CTX, the faster the chunk (linear, about 1.63 ms per 96 positions per 4 layers), but the memory is shorter.
- lm_head and token selection run on the CPU (not part of the chunks); model.norm in chunk 26-28 with PART 0 is not applied (the norm runs on the CPU). lm_head int8: one script prepares lm_q.bin and lm_s.bin, a second prepares the bf16 embedding and nw.bin.

## 6. Checking on the phone (test stand, FACT)
- Stand: package com.enntest.app (the test app, sources not included). Installing with a different signing key gives INSTALL_FAILED_UPDATE_INCOMPATIBLE: run adb uninstall first. Always use adb with -s localhost:5555 (if there is no connection, reconnect adb).
- Which build is for what (install: adb -s localhost:5555 install -r FILE):
  - single-exec build: executes exactly one file files/p.nnc with input files/in.bin and output files/out.bin (needed by the single-chunk, chain and multi-layer checkers);
  - bench build: opens b00-b13 and writes files/bench.txt (needed by the bench scripts);
  - generation-loop builds (three iterations, the last is final): §7.
  With the wrong build installed, out.bin does not appear (bench build) or tok/s is 0.0 (an earlier loop build instead of the final one).
- Chunks: the files go to ~/npu_out/out (13 files of 63791104 B each; transfer as an archive: tar in the cloud, Download on the phone, tar xf into ~/npu_out).
- Single-chunk check N: builds the chunk input from the CPU chain (21 prompt tokens and a test token id 13), deletes the old out.bin on the phone, copies L{a}e{a+2}.nnc to files/p.nnc, waits for out.bin up to 80 s, computes cosine over 5 parts against onnx. Run it for one chunk or for all 13.
- Chain check: the 14-chunk chain on the NPU (input of chunk N = output of chunk N-1).
- Multi-layer checks: 4 layers (hardcoded); an arbitrary start layer and count, e.g. 2 8 for L2e10 or 2 14 for L2e16 (1+2N cosines); the same for float16 by name (L2e10h_tm); and by index K (0, 1, 2 = L0e2h_tm, L2e16h_tm, L16e28h_tm): cosine, minimum over parts, max hout; it waits for out.bin up to 300 s.
- Expected: cosine 1.0 on all parts, and max hout of the NPU and ONNX agree to tenths (FP16 rounding).

## 7. The generation loop in C and measurements (FACT)
- Build (shown for the final loop build; file names are yours to choose): clang -O3 -march=armv8.2-a+dotprod -shared -fPIC -o loop.so loop_app.c -llog -ldl -lm; cp loop.so lib/arm64-v8a/libenntest.so; cp base.apk unsigned.apk; python3 pack.py; zipalign -f -p 4 unsigned.apk aligned.apk; apksigner sign --ks KEYSTORE.jks --ks-key-alias ALIAS --ks-pass env:KS_PASS --out signed.apk aligned.apk; adb -s localhost:5555 install -r signed.apk. pack.py is a local packaging step. The key password is passed through KS_PASS, kept outside the repository (for example in ~/keys.env; load it with set -a; . ~/keys.env; set +a). A lost keystore cannot be recovered; keep a backup.
- Files in files/ on the phone: three .nnc (L0e2h_tm, L2e16h_tm, L16e28h_tm), lm_q.bin and lm_s.bin, top5.bin, c20/emb.bin and c20/nw.bin; the reference comes from a CPU reference run. About 0.9 GB of memory is needed for the chunks inside the app.
- generation_runner N TF (N up to 80; TF 0 free generation, 1 feed the reference tokens): writes the config, waits for files/run20.txt (the last line is END), prints mismatches and RESULT. Inside the app: 21 prompt tokens, 80 steps, top1 compared with top5.bin.
- The second loop build has its own config script (N, TF, run seconds SUS, mode MD: 0 normal, 1 without lm_head, 2 without running the NPU). After that script you MUST install the final loop build. In modes 1 and 2 "tok/s" means loop passes; RESULT in mode 2 shows no accuracy.
- device_bench R1 R2 SUS [nopush] (defaults 100 30 0): pushes b00 (chunk 0-1 from Download) and b01-b13 (L2e4 ... L26e28 from ~/npu_out/out), in_t.bin and cfg.txt; nopush skips the upload (the upload needs 2.5 GB of free space). To time one chunk: adb push it as files/b01.nnc, run device_bench 100 30 0 nopush, read "chain piece 1 avg exec", then restore b01 from L2e4_fp16.nnc. device_ab_bench NAME1 NAME2 ...: a series of such measurements with b01 restored at the end (synchronous).
- Before every measurement: ls -l files/b*.nnc (14 files of 63791104 B) and a check of the installed APK with preflight_check. Measure only while charging from 50 percent, one chunk at a time, and compare only within one series (noise between series is up to 0.3 ms).
- Energy: only energy_charge_counter (Charge counter, about 22 min, not charging, screen on, needs the final loop build): idle, NPU, GPU F16, NPU, GPU F16, idle; the counter step is 4290 uAh (resolution 0.38 W); if a row has fewer than 5 different counter values, the method is not usable. The older scripts that read current from dumpsys battery are noisy by a factor of 1.5-2: do not use their numbers (LOG sec. 3).
- llama.cpp for comparison: ~/llama.cpp/build/bin/llama-bench (Vulkan), models in ~/gguf (Qwen3-0.6B f16, Q8_0, Q4_K_M). CPU: cd ~/gguf; llama-bench -m FILES -ngl 0 -t 1,2,4,6,8 -p 0 -n 64 -r 3; GPU (Xclipse 960): -ngl 99 -t 4 -p 128 -n 64 -r 3.

## 8. Errors and workarounds (FACT)
| Symptom | Cause | Workaround |
|---|---|---|
| OSError: Background processes not supported | Jupyter does not accept & | subprocess.Popen, output to a file |
| eais init: workspace is not valid directory | the folder does not exist | mkdir -p |
| eais init: files with model name does not exist | no w_NAME.onnx | name the file like the folder |
| eais config: TypeError CommentedMap | a bug in eais output | read the configs with cat |
| generate_tv: RMSNORMALIZATION is not supported by ICMI | the CPU simulator does not know the node | ignore, return code 1 is expected |
| eais compile: model needs to be converted | no record of the conversion | compiler.compile from Python (§4.5) |
| TypeError NoneType not subscriptable (graphgen.py) | a list in input_model was expanded | use a dictionary |
| Get axis failed, type must be number | axis is an array | fix_axis (§4.3) |
| CDI getHandler failed, Type RmsNorm | gamma is [1,1,1,C] | fix_gamma (§4.4) |
| AdvProfiler: Reshape [1,1,1,1024] invalid for size 1048576 | the same gamma, with A8W8 and A16W8 | §9 |
| float16: onnxruntime error on OutportReshape | conversion with a float16 interface | expected, continue with fix_axis, fix_gamma, compile |
| Compile: DataFormat not connected | the old web-service path | Mul 1.0001 on x |
| Blob format inconsistency | web service, 4 or more layers | not reproduced in the cloud |
| device check: adb missing | wireless debugging is off | reconnect adb |
| device check: out.bin not deleted or not produced | file permissions; the chunk did not open within 80 s | check the size of files/p.nnc, check which build is installed, run the chunk alone |
| bench: tok/s 0.0 or no run file | the wrong build is installed | install the final loop build |
| cat > file overwrote a script | no check | test ! -e FILE before cat >; back up first |
| The phone switches off when a .nnc runs | L2e10q_tm (A16W8 tuning_mode), 3 of 3 | do not run it, cause not found |

## 9. A16W8 and quantization (workaround, FACT; low value)
- Result: no speed gain (LOG sec. 2); not tried with real calibration; accuracy on random calibration is poor. You only need this if you return to quantization.
- Build (three scripts: a16w8_init, a16w8_quantize, a16w8_compile; the first two are written for chunk L2e10, adapt the names): eais init, activation_bitwidth 16 (bw_kernel 8, quantize_type asymm); the standard conversion fails at AdvProfiler. Workaround: put the RMSNorm gamma into [1,C,1,1] in Converter_result/convert/w_NAME_new_snc.snc, do NOT touch the axis (the quantizer needs a list), save fixed_in.snc; run the quantizer directly: PYTHONPATH=.../eais/lib/ENNQuantizer, cwd is the model folder (database.txt with a relative path, otherwise No data to profile model): eais_conversion.py --snc_file fixed_in.snc --out_fdr Q_out --preprocess DimensionalShift MergeConcat AdvProfiler --quantizer AsymQuant --calibration_method percentile --profiler_select AdvProfiler --postprocess OutportReshape --target_soc THETIS --bit_width_scale 12 --dataset Config/DATA/database.txt --iteration 2 --dtype_edge INT16 --dtype_weight INT8 --input_dtypes FLOAT32 --output_dtypes FLOAT32; in Q_out/HW_fixed_in.snc set the axis to a number, then compile as in §4.5.
- Multicore: in w_NAME.yaml set compiler mode multicore, compile the same HW_axis.snc into a new folder (flag -D multicore, multicore=true;num_cu_cores=8); the file name still says SingleCore, so judge by measurement. Gain on chunk 0: 2.19 ms against 2.67 (A16W8) and 2.97 (FP16). Configs: eais/asset/config/compiler_default_config.yaml, compiler_multicore_config.yaml, op_device_config.yaml.

## 10. RMSNorm in SNC: what is known
- FACT: Reshape [1,1024,1,1] (NPU, FP16) -> RMSNormalization (device CU, axis [1], epsilon 1e-6, stash_type 1) -> Reshape [1,1,1,1024]. A layer has one internal tensor, kernel (gamma). A two-layer chunk has 8 RMSNorm nodes. The channels are on axis 1, gamma lay on axis 3.
- INFERENCE: the cause of both compiler errors and of the AdvProfiler failure is the gamma shape: only it was changed (fixed.snc failed, fixed2.snc passed). There is no direct A8W8 comparison after the fix.
- THEORY: the newer optimizer version (cloud 8.16.21.25 against 8.15.5.20 in the service) assembles RMSNorm differently from Pow, ReduceMean, Sqrt, Div(1, Sqrt), Mul.

## 11. Profiling and what does not work (FACT)
- eais nnc-reader gives only the header (SoC 9965, NPUC v2.4.11.o, the compiler command). NNCInfo.csv and back_end_*.csv carry no timings. VisualProfiler does not fit (no Thetis, no STMGeneration, STMAnalyzer, STMAnnotation folders, and it needs SSH to the phone). Per-node timing comes only from experiments on the phone.
- The graph profile: ~/w_NAME/Compile_out/gen_files/graph_0/npu/backend/profile_log.txt: the only executors are GNPU and SDMA (about 40 percent of nodes are SDMA); attention is split into 8 GNPU nodes by KV head, preceded by SDMA permute_to_CHWN and kernel_reshape.toKCell.
- Launching .nnc from adb shell without root is blocked by sepolicy; from an app it works. NPU interrupt counters and devfreq are not usable for measurement without root. There is no public step-by-step LLM instruction for eais.

## 12. Not checked
- One chunk for all 28 layers, variant E in three chunks, context 32 or 64 in three chunks, A16W8 and A16W4 with real calibration, NPU together with GPU, long runs, Exynos Lab session limits. Full list: LOG sec. 7.
