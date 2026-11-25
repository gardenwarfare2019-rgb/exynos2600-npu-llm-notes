# Running a self-built Qwen3-0.6B on the Exynos 2600 NPU

Unofficial notes: how to build your own language model (Qwen3-0.6B, FP16) and run it on the Exynos 2600 NPU with Samsung's cloud SDK (`eais`, Exynos Lab), then verify it on a phone. A step-by-step instruction for LLMs with `eais` was not publicly available, so this repository records the working path, workarounds for the errors, measurements, and the dead ends.

**Short version:** the model works on the NPU as three FP16 chunks and every chunk is accurate (cosine 1.0 against ONNX). It reaches 23-30 tokens/s. On this phone the GPU (llama.cpp, Vulkan) is faster and uses less energy per token, so the NPU is not the better choice for chat. Details and numbers are in the log.

## Contents
- `NPU_GUIDE.md`: the recipe (building an FP16 chunk, writing ONNX for the NPU, checking on the phone, the C generation loop, a table of errors and workarounds).
- `LLAMA-ALT-01_LOG.md`: the log with numbers, dead ends, the result and what was not tried.

- `scripts/cloud/`: the scripts that run in the Exynos Lab notebook (build a chunk, fix the graph, compile, the A16W8 workaround). Small and commented.
- `examples/`: `NNCInfo_*.csv` of three compiled chunks (input and output sizes, FLOAT16).

The phone-side scripts and the Android test app are not included. `NPU_GUIDE.md` section 0 describes each of them by role (what it takes, what it does), which is enough to rewrite them. The raw SDK logs of the builds are not included either.

## Confidence labels
- **FACT**: there is output or a file, a date and a source.
- **INFERENCE**: follows from facts by arithmetic or logic.
- **THEORY**: an explanation that has not been checked.
- **HYPOTHESIS**: there is a planned check.
- **REFUTED**: checked and found wrong (with a date).
- **RETRACTED**: the author or assistant made a mistake, the conclusion was withdrawn (with the reason).
- **REPORTED**: stated by the author, the output was not seen.

Refuted and retracted claims are kept in the text on purpose: dead ends are as useful as the working path.

## Please note
- **Risk.** One of the built files (`L2e10q_tm`, A16W8 in tuning_mode) switched the phone off 3 times out of 3; the cause was not found. Everything here is at your own risk.
- **Scope.** Results come from one phone and one SDK version (`eais` 2026.4.21). Other devices and versions may differ.
- **Not affiliated with Samsung.** These are independent notes. Product names and tools belong to their owners. No Samsung models, binaries or catalog files (including catalog `.nnc` files) are included.
- **Secrets.** Signing keys and passwords are never committed: the key password is passed through the `KS_PASS` environment variable.

## License
MIT, see the `LICENSE` file. It covers the text and the scripts.
