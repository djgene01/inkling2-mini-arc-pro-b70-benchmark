# Inkling 2 Mini on vLLM XPU for one Arc Pro B70, following SergiioB/intel-arc-pro-b70-inference-cookbook
# docs/inkling2-mini/INKLING2-MINI-VLLM-XPU.md (upstream d1e3506, 2026-11-22): vLLM 0.34.0rc1.dev1133 XPU wheel,
# mgaruccio/vllm-inkling2-xpu@2e9d28b0, vLLM PR #69186 patch, AOT bmg_g31 NVFP4 CUTLASS Xe2 MoE + SYCL M=1 decode
# attention, XPU vision tower + FP8 KV. All extras are opt-in at run time (INK2_CUTLASS_MOE, INK2_SYCL_DECODE_ATTN).
#
# Three stages: "base" (Python packages) keeps the image's own GPU runtime (compute-runtime 26.18, Level Zero 1.28.2),
# which PyTorch 2.15+xpu runs on. "build" adds ocloc for the AOT kernels, but ocloc pulls compute-runtime 26.40, under
# which PyTorch 2.15 segfaults on the first device query; so the final image is "base" plus the built venv only.
FROM intel/oneapi-toolkit@sha256:f5fe00835cc363089190ac9855f290c2315f87320e0973bc36a740663cdaf96a AS base

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=8 UV_CONCURRENT_DOWNLOADS=4
RUN curl -fsSL -o /tmp/uv.tgz https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz \
 && tar xzf /tmp/uv.tgz -C /tmp && install /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv && rm -rf /tmp/uv*

# Python packages: run with the oneAPI library paths cleared so nothing links against the toolkit by accident.
RUN env -u LD_LIBRARY_PATH uv venv --python 3.14 /opt/venv \
 && env -u LD_LIBRARY_PATH uv pip install --no-build --python /opt/venv/bin/python \
      --index https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match \
      --find-links https://vllm-project.github.io/vllm-xpu-kernels/whl/vllm-xpu-kernels/ \
      "https://wheels.vllm.ai/647a76866b4559a74e27d1c6b55fbffe85c07ce4/vllm-0.34.0rc1.dev1133%2Bg647a76866.xpu-cp38-abi3-manylinux_2_34_x86_64.whl" \
      "torch==2.15.0+xpu" \
      "triton @ https://wheels.vllm.ai/647a76866b4559a74e27d1c6b55fbffe85c07ce4/triton-3.9.0%2Bxpu-py3-none-any.whl" \
      "vllm-xpu-kernels==0.1.17" \
 && env -u LD_LIBRARY_PATH uv pip install --python /opt/venv/bin/python --no-deps \
      "git+https://github.com/mgaruccio/vllm-inkling2-xpu@2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9" \
 && rm -rf /tmp/uv-cache

COPY patches /opt/inkling-patches

# Required: vLLM PR #69186 (FP32 lm_head on XPU without recasting the BF16 weight every step).
RUN LP=$(env -u LD_LIBRARY_PATH /opt/venv/bin/python -c "import inspect, vllm.model_executor.layers.logits_processor as m; print(inspect.getfile(m))") \
 && patch -p1 -d "$(dirname "$(dirname "$(dirname "$(dirname "$LP")")")")" < /opt/inkling-patches/pr69186-logits-processor.patch

# Header checkouts for the CUTLASS Xe2 kernel, pinned to Sergio's commits.
RUN for spec in "intel/sycl-tla 7b56b936b87f799f80dde60c2fd8d11c1128cebb /opt/src/sycl-tla" \
                "vllm-project/vllm-xpu-kernels e9b6e4b24a69886a8b384edc06e7a631ad282705 /opt/src/vllm-xpu-kernels"; do \
      set -- $spec; git init -q $3 && git -C $3 fetch -q --depth 1 https://github.com/$1.git $2 && git -C $3 checkout -q FETCH_HEAD; done

FROM base AS build
# ocloc turns the SYCL kernels into bmg_g31 machine code (AOT); the toolkit image lacks it. It and the GPU driver come
# from the image's own kobuk-team intel-graphics PPA, at one matching version (build stage only, see the top).
RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq intel-ocloc libze-intel-gpu1 intel-opencl-icd \
 && rm -rf /var/lib/apt/lists/* && ocloc --version 2>&1 | head -2 || true

# icpx 2026.0 does not enable the SPIR-V extensions the CUTLASS Xe2 code uses; pass them as vllm-xpu-kernels' CMake does.
RUN sed -i 's#-O3 -std=c++20 -fPIC -shared \\#-O3 -std=c++20 -fPIC -shared -flink-huge-device-code -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \\#' \
      /opt/inkling-patches/build-nvfp4-cutlass-xe2.sh /opt/inkling-patches/build-sycl-decode-attn-xe2.sh \
 && grep -c "SPV_INTEL_split_barrier" /opt/inkling-patches/build-nvfp4-cutlass-xe2.sh /opt/inkling-patches/build-sycl-decode-attn-xe2.sh \
 && sed -i 's#-DCUTLASS_VERSIONS_GENERATED \\#-DCUTLASS_VERSIONS_GENERATED -DTORCH_EXTENSION_NAME=_inkling_nvfp4_xe2 \\#' \
      /opt/inkling-patches/build-nvfp4-cutlass-xe2.sh \
 && grep -q "TORCH_EXTENSION_NAME=_inkling_nvfp4_xe2" /opt/inkling-patches/build-nvfp4-cutlass-xe2.sh \
 && sed -i 's/^@@ -117,6 +136,14 @@ class Inkling2Experts/@@ -117,6 +136,13 @@ class Inkling2Experts/' \
      /opt/inkling-patches/vllm-inkling2-xpu-cutlass-optin.patch \
 && grep -q "^@@ -117,6 +136,13 @@" /opt/inkling-patches/vllm-inkling2-xpu-cutlass-optin.patch

# Opt-in kernels and features, in Sergio's APPLY.md order.
# nvfp4_xe2_ext.cpp includes "csrc/sycl_first.h" from the vllm-xpu-kernels root, which the build script does not add.
# icpx needs the oneAPI environment, but PyTorch must load its own newer Intel runtime from /opt/venv/lib,
# not the toolkit's older libur_loader, so the venv libraries go first on the path.
# Include folders as vllm-xpu-kernels' own CMake sets them for sycl-tla (include, tools/util/include, applications).
RUN export LD_LIBRARY_PATH=/opt/venv/lib:$LD_LIBRARY_PATH \
      CPLUS_INCLUDE_PATH=/opt/src/vllm-xpu-kernels:/opt/src/sycl-tla/tools/util/include:/opt/src/sycl-tla/applications:$CPLUS_INCLUDE_PATH \
 && VENV=/opt/venv SYCL_TLA_DIR=/opt/src/sycl-tla VLLM_XPU_KERNELS_DIR=/opt/src/vllm-xpu-kernels \
      bash /opt/inkling-patches/build-nvfp4-cutlass-xe2.sh \
 && VENV=/opt/venv bash /opt/inkling-patches/install-vision-fp8kv-xpu.sh \
 && VENV=/opt/venv bash /opt/inkling-patches/build-sycl-decode-attn-xe2.sh \
 && rm -rf /tmp/inkling-*-build

FROM base
COPY --from=build /opt/venv /opt/venv

# Run time: the venv first, and PyTorch's own Intel runtime instead of the toolkit's (GPU driver libs stay in /usr/lib).
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH= \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_RPC_BASE_PATH=/tmp/ink2-ipc
RUN mkdir -p /tmp/ink2-ipc && python -c "import vllm, vllm_inkling2_xpu, vllm_inkling2_xpu.ops._inkling_nvfp4_xe2, vllm_inkling2_xpu.ops._inkling_attn_xe2; print('vllm', vllm.__version__, '+ plugin and both kernels import')"
