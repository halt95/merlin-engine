# Merlin 3.0.1: build this checkout on vLLM 0.32.0.
# unqualified until qualified on hardware
# Build from the repository root, using the source ref as the image tag:
#   SOURCE_REF=merlin-v3.0.1
#   docker build -f container/Dockerfile --build-arg SOURCE_REF="$SOURCE_REF" -t "merlin-engine:$SOURCE_REF" .
# Base index digest: Docker Hub library/python, 3.13-slim-trixie, read 2026-10-15.
FROM python:3.13-slim-trixie@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
ARG UV_VERSION=0.12.13
ARG SOURCE_URL=https://github.com/halt95/merlin-engine.git
ARG SOURCE_REF=merlin-v3.0.1
ARG MAX_JOBS=4
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 UV_PYTHON_DOWNLOADS=never \
    CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14 \
    CUDA_HOME=/opt/venv/lib/python3.13/site-packages/nvidia/cu13
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates tar gzip procps build-essential gcc-14 g++-14 ninja-build \
      cmake pkg-config libnuma-dev libssl-dev patch util-linux \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -fsSL \
      "https://releases.astral.sh/github/uv/releases/download/$UV_VERSION/uv-installer.sh" \
      -o /tmp/uv-installer.sh \
    && UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/uv-installer.sh \
    && rm /tmp/uv-installer.sh
WORKDIR /opt/merlin
COPY . /opt/merlin/
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
ENV PATH="/opt/cargo/bin:$PATH"
# Match the checkout toolchain instead of the distribution's Rust packages.
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain none \
    && rm /tmp/rustup-init.sh \
    && rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"$/\1/p' rust-toolchain.toml)" --profile minimal
# The shared installer owns the 0.32 lock, CUDA toolkit, native build and FlashInfer tile patch.
# Incomplete pins or missing tile tooling must fail the build; there is no legacy bundle fallback.
# Fetch public source history to verify the copied checkout and its vLLM 0.32.0 ancestry.
# Besides tracked files, the check fails on ignored build products (native libraries, rust/target,
# vllm/_version.py, .build-products*) that a previously built checkout can leave in the context.
# The host's private Git metadata is excluded from the build context.
RUN git init --quiet \
    && git config core.filemode false \
    && git fetch --quiet --no-tags "$SOURCE_URL" "$SOURCE_REF" \
    && git reset --mixed --quiet FETCH_HEAD \
    && git diff --exit-code HEAD -- \
    && ! git ls-files --others --ignored --exclude-standard | grep -E '\.(so|o|a)(\.|$)|^vllm/_version\.py$|^vllm/vllm-rs|(^|/)target/|(^|/)\.build-products' \
    && test "$(gcc-14 -dumpversion | cut -d. -f1)" -eq 14 \
    && test "$(g++-14 -dumpversion | cut -d. -f1)" -eq 14 \
    && chmod +x release/*.sh models/*/serve*.sh container/*.sh \
    && MAX_JOBS="$MAX_JOBS" CARGO_BUILD_JOBS="$MAX_JOBS" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
       VLLM_TARGET_DEVICE=cuda \
       bash release/install-env.sh /opt/venv \
    && /opt/venv/bin/python -c 'from importlib.metadata import version; from packaging.version import Version; assert Version(version("vllm")).base_version == "0.32.0" and Version(version("vllm")).pre is None, version("vllm"); assert version("flashinfer-python") == "0.7.0.post1"; assert version("humming-kernels") == "0.1.17"; assert Version(version("instanttensor")) >= Version("0.2.2")' \
    && rm -rf "$HOME/.cache"
ENV TREE=/opt/merlin VENV=/opt/venv HOST=0.0.0.0 PORT=8000 CACHE_ROOT=/cache \
    MERLIN_MODEL=flash-next CHECKPOINT_DIR=/models/checkpoint \
    IMG_CAP_PX=8388608 MAX_PIXELS=8388608 \
    FLASHINFER_WORKSPACE_BASE=/cache/flashinfer TRITON_CACHE_DIR=/cache/triton
ENV PATH="/opt/venv/bin:/opt/venv/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH"
VOLUME /cache
EXPOSE 8000
# A generation exercises the engine; /v1/models alone can outlive a dead engine.
HEALTHCHECK --interval=60s --timeout=90s --start-period=30m --retries=3 \
  CMD ["/opt/venv/bin/python", "/opt/merlin/container/healthcheck.py"]
ENTRYPOINT ["bash", "/opt/merlin/container/docker-entrypoint.sh"]
