# LLAMA-ALT-01: log (NPU Exynos 2600)
Closed on 23 Nov, can be reopened. Recipes and commands: NPU_GUIDE.md. Labels: see README (FACT, INFERENCE, THEORY, HYPOTHESIS, REFUTED, RETRACTED, REPORTED). Dates are 2026.

## 1. Result
- FACT: our own Qwen3-0.6B in FP16, in three chunks (layers 0-1, 2-15, 16-27), runs on the NPU and every chunk is accurate (cosine 1.0). The generation loop is in C and needs no adb. Teacher forcing: 78 of 80 tokens. Free generation: 55 of 80, the first 35 in a row equal the reference, the first divergence is at t=56 (logit gap 0.05, the app's token is second in the reference).
- FACT: speed 23-30 tok/s (a one-minute run: 33.3 ms per token = NPU 2.3 + 14.7 + 12.6, lm_head int8 on the CPU 3.8). llama.cpp on the GPU: F16 40-48, Q8_0 80, Q4_K_M 115 tok/s.
- INFERENCE: the NPU is worse than the GPU in both speed and energy (225-233 against 116-142 mJ per token, Charge counter, 2 runs each, method resolution 0.38 W). For chat the NPU is not the better choice.
- Not checked and not done: section 7. Dropped: root (gives no speed), gen-ai (unavailable in the cloud), LiteRT (author's decision).

## 2. Key numbers
| What | Value | Label |
|---|---|---|
| Chunk 0 (2 layers) alone, FP16 | 2.97 ms (cold 7.0) | FACT |
| A chunk in a chain by layer count | 2: 4.5; 4: 7.15; 8: 12.40 (tuning_mode 11.33); 14: 20.04 ms | FACT |
| Chunk model | about 1.9 ms per launch + 1.27-1.31 ms per layer | INFERENCE (3-4 points) |
| 14-chunk chain | 62.3 ms, 16 cycles/s | FACT |
| Three chunks in the C loop | 33.3-42 ms per token | FACT |
| One chunk for 28 layers | about 33 tok/s | THEORY (extrapolation) |
| CPU onnxruntime, 14 chunks | 94 ms (4 threads, best) | FACT |
| lm_head | fp32 18.5 ms; int8 sdot 3.21 ms alone, 3.8-7.5 inside the app (cause not found) | FACT |
| lm_head int8 accuracy | top1 80 of 80, top5 set 70 of 80 | FACT |
| embed_tokens and lm_head | byte-identical | FACT |
| llama.cpp CPU, tok/s (best) | F16 41.9, Q8_0 66.1, Q4_K_M 68.7 | FACT |
| llama.cpp GPU tg64 / pp128 | F16 48.4 / 1591; Q8_0 80.2 / 1349; Q4_K_M 114.6 / 1368 | FACT |
| Weight read bandwidth | GPU about 72 GB/s, CPU 34, NPU 14.5 (chain), about 22 (L2e16) | INFERENCE |
| A16W8 against FP16 | L2e10: 11.14 against 10.88 ms, no gain; on 2 layers 2.67 against 2.97 | FACT |
| float16 interface (L2e10h) | 10.3 against 10.9 ms (6 percent), on the edge of noise 0.3-0.7 | FACT |
| tuning_mode | about minus 9 percent (0.65 ms on 4 layers, 11.33 against 12.40 on 8) | FACT |
| Noise between measurement series | up to 0.3 ms, the control drifts up to 0.9 | FACT |

## 3. Energy
- FACT, Charge counter (energy_charge_counter, not charging, screen on, 160 s window): idle 0.57 W (rough); NPU 24.2 tok/s, 5.45 W above idle, 225 mJ per token; GPU F16 27.6 tok/s, 3.19 W, 116; NPU 23.3 tok/s, 5.41 W, 233; GPU F16 32.5 tok/s, 4.62 W, 142. The counter step is 4290 uAh, resolution 0.38 W. The speed of both dropped (heating to 33-44 C), cause THEORY (throttling).
- RETRACTED 23 Nov: all measurements through the current from dumpsys: 99 against 97 mJ (NPU and GPU "equal"), 154 against 156, 133/193/91, 304/129. Repeats of the same mode differ by 1.5-2 times, GPU power wandered 4.57-9.09 W at the same speed; these numbers must not be compared. Only this survives: without lm_head the loop is 4 ms faster.
- RETRACTED 22 Nov: the "after lunch" runs (12.9 cycles/s, 242-302 mJ): slot b01 held chunk L2e16 instead of L2e4, so the chain computed 40 layers.
- HYPOTHESIS, not checked: the share of the CPU (lm_head, waiting for exec) in the NPU-mode power; NPU together with GPU.

## 4. Timeline and dead ends (one line each)
Samsung web service (21 Nov, package 8.15.5.20, A8W8 only, random calibration 0..1):
1. Dynamic shapes: Init fails; static shapes: Conversion fails on SimplifiedLayerNorm.
2. torch.onnx.export does not work in Termux: ONNX is written by hand through protobuf, RMSNorm as Div(1, Sqrt), opset 17.
3. Whole model: calibration is killed (memory). Halves of 14 layers: Conversion Failed with a clean log. Several inputs and outputs: Failed. Workaround: one input inp and one output out.
4. Compile: DataFormat not connected. Workaround: Mul 1.0001 on x (the factor is about 1.0014 over 14 chunks). Why it helps: THEORY.
5. Blob format inconsistency: 1 layer passes, 4 or more fail, 2-3 randomly (a graph-size threshold). Decision: 14 chunks of 2 layers, re-upload the failed one.
6. A8W8 accuracy on real data: cosine 0.04-0.78 (on the service's data 0.98-0.9998). INFERENCE: the ranges are set by a calibration on positive values 0..1, real values with a minus sign get clipped; factors on input and output do not cure it. Zeros in nv: one scale per output, step about 1.08.
Exynos Lab cloud (22 Nov, eais 8.16.21.25):
7. FP16: conversion fails at the last check (the ICMI simulator does not know RMSNormalization), the SNC files are written. FACT.
8. eais compile refuses: workaround through compiler.compile from Python with input_model as a dictionary.
9. Compiler error: RMSNorm axis as an array. Fixed through snc_py_api: a number.
10. CDI getHandler error on RmsNorm: gamma [1,1,1,C] with channels on axis 1. Rewritten as [1,C,1,1]: compile passed, .nnc 63791104 B. INFERENCE: the cause is the gamma shape (only it was changed, fixed.snc failed, fixed2.snc passed).
11. FP16 on the device: 14 of 14 chunks cosine 1.0, the 14-chunk chain on the NPU 1.0, the error does not accumulate (token 21).
Other:
12. L2e10q_tm (A16W8 tuning_mode, 130994176 B, md5 6eef99cd444a5599e961e74451bca654) switches the phone off 3 times out of 3. FACT. Cause not found (THEORY: an NPU or driver failure on this .nnc; there are no traces in kmsg, that is the bootloader log of the next boot). RETRACTED 23 Nov: "a hardware power failure" (the voltage drop happened after the reboot). L2e10tm and L2e10q do not switch it off. Do not run this file.
13. A16W8 only with random calibration: accuracy is poor (cosine 0.14, 0.54, 0.76, -0.45, 0.19 by parts), no speed gain. Real calibration was not done.
14. gen-ai: unavailable in the cloud (the config points to /lib/compiler_GenAI, the folder does not exist). model_type LLM and skip_4_dim_conversion: the graph does not change (md5 of profile_log is the same, legacy does not read these keys).
15. REFUTED 22 Nov (profile_log of L2e10: the only executors are GNPU and SDMA): SOFTMAX, CONCAT and PAD are not on the GPU, assign_npu is not needed.
16. Dependence on context: a 4-layer chunk with CTX 32/64/128 = 5.22/5.86/6.84 ms, slope about 1.63 ms per 96 positions (FACT). Their graph is the same (880 lines, toKCell 256, permute 128): the data volume grows, not the node count. Cause: HYPOTHESIS (many small SDMA transfers, one per 8 KV heads).
17. Attention variants A (K transposed in advance), B (no Concat), C (A+B): the time does not change (7.17/7.01/7.07 against 7.16/7.06). H2 (software cast FP32-FP16) REFUTED: the slope in tuning_mode is the same (1.64).
18. Variant E (Mul + ReduceSum instead of MatMul): on the CPU it matched (maxdiff 1.5e-05), on the NPU L2e6 cosine 1.0, faster than the control by 1.2 ms at CTX 128 (5.49 against 6.70, two series, spread 0.68). Dropped for 14 layers (a gain of about 4 out of 20 ms), not tried in three chunks.
19. Anomaly: the energy loop ran 25 cycles/s against 14.4 for the bench script with the same chunks. THEORY: depends on the previous load (boost). The check (bench 1 1 60, then bench 100 30 60) was not done. The C loop gave 29-30.
20. Search for guides: there is no step-by-step LLM instruction for eais (Samsung's blog is an overview). Open sources: the ExecuTorch Exynos backend (FP16, INT8, INT16, build from source), ultralytics #609 (LiteRT on Exynos 2600, FP16 and INT4 rejected).

## 5. The model
- Weights: ~/qwen3 (safetensors 1503300328 B, BF16, 28 layers, hidden 1024, 16 heads, 8 KV heads, head_dim 128, rope_theta 1e6, eps 1e-6, vocab 151936).

## 6. Accuracy tests by layers (FACT, token 21 and the loop)
- All 14 two-layer chunks: cosine 1.0 on 5 parts, max hout of the NPU and ONNX agree (from 6.2 to 313.8), no FP16 overflow.
- Chunks of 4, 8 and 14 layers (L2e6, L2e10, L2e16, also tuning_mode): 1.0 on 9, 17, 29 parts. Three chunks: L0e2h_tm 1.0 on 5, L2e16h_tm on 29 (max 35.8/35.8), L16e28h_tm on 25 (297.0/297.5).
- The CPU chain 0-1, 2-15, 16-27 against the 14 chunks: top1 80 of 80, max hidden difference 0.054.
- model.norm in chunk 26-28 with PART 0 is not applied, the norm runs on the CPU (nw.bin).
- Not checked: exact match with FP32 logits, runs longer than 80 tokens (the cache is 127).

## 7. Not tried (ideas for a return)
- One chunk for 28 layers (first verify the graph of layers 0-1: it was built by a separate script).
- Variant E in three chunks; context 32 or 64 (faster, but shorter memory).
- A16W8 and A16W4 with real calibration (about 8.3 MB per sample; a risk of switch-off like L2e10q_tm), smoothquant, adaround, mpq_name_dict, model_type LLM for quantized builds, general.mode gen-ai-eht.
- NPU in parallel with GPU (1.34-6x in papers, THEORY: both read the same memory), prefill on the NPU, other models.
- The energy method on a longer window and with a charging control; the CPU share in the NPU-loop power.

## 8. State of the stand on 23 Nov
- Phone: the final loop build installed; files/b00-b13 of 63791104 B each (FP16, for the bench), three loop chunks L0e2h_tm, L2e16h_tm, L16e28h_tm (64020480 / 447500288 / 383590400 B; md5 2ae257f1f0219dbe766dd8a2b4c429b0, 9fd07406022b80c8846267b6838c8b75, 9f1fba744e232cb05bdabbdaff8985c0). Download holds L2e10tm.nnc, L2e10q.nnc, L2e10q_tm.nnc (do not run the last). Copies are in ~/npu_q.
- Exynos Lab cloud: the session is temporary, consider it lost; the scripts were saved separately.
- Which build to use: the bench scripts need the bench build; the chunk checkers need the single-exec build; the loop and the energy scripts need the final loop build. Before bench always run ls -l files/b*.nnc (14 files of 63791104 B).
- Measure only while charging from 50 percent, one chunk at a time; the A/B bench script is synchronous.

## 9. Process mistakes and safeguards
- Mistakes: b01 held L2e16 instead of L2e4 (22 Nov); an earlier loop build installed instead of the final one (23 Nov); a shell redirect overwrote an existing script (restored from its .pyc; rule: test ! -e before cat >); the self-check in the fixed-range generator was dead (an early sys.exit(0); equivalence is verified by a separate script); an extra chunk in a benchmark filter (fixed in the second version of the script); a generator range 14-27 was misread as 14-16 (the extra argument 0 is mandatory); grep with head -12 missed records; a wrong dumpsys dropbox tag; the first version of a multi-layer checker used a zero cache (19.4 instead of 17.2).
- Safeguards: a preflight check (sizes of b*.nnc and whether the installed APK is the expected one, before every measurement), the test ! -e rule before cat > file, confirming the build against the table in section 8.

## 10. Integration (discussion on 23 Nov)
- Result: an Android app without UI, the prompt is hardcoded (21 tokens from the reference script), the tokenizer exists only in Python, token selection is greedy, context 127. Files are about 1.4 GB (chunks 0.9 GB, embedding 311 MB, lm_head int8 155 MB). Exynos 2600 only; the signing key is a test key (the password is passed through KS_PASS).
- Sensible uses: background tasks next to a busy GPU (THEORY) and small non-language models (THEORY). For a concrete application a task, a place and a way to call the local model are needed.
