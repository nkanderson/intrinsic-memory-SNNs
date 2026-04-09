# Multi-stage Dockerfile for cocotb testing with SystemVerilog
# Uses oss-cad-suite which includes Verilator and other EDA tools

FROM ubuntu:22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
  curl \
  wget \
  git \
  make \
  build-essential \
  python3 \
  python3-pip \
  python3-venv \
  libffi-dev \
  && rm -rf /var/lib/apt/lists/*

# Stage for oss-cad-suite installation
FROM base AS oss-cad-suite

# Install oss-cad-suite (includes Verilator, Yosys, cocotb, and other tools)
# This layer is cached unless the version changes
# Note: The tarball filename uses format YYYYMMDD (no dashes) while the release tag uses YYYY-MM-DD
ARG OSS_CAD_SUITE_VERSION=2026-01-14
ARG OSS_CAD_SUITE_DATE=20260114
RUN wget -q https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE_VERSION}/oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz \
  && tar -xzf oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz -C /opt \
  && rm oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz

# Final stage
FROM base AS final

# Copy oss-cad-suite from installer stage
COPY --from=oss-cad-suite /opt/oss-cad-suite /opt/oss-cad-suite

# Add oss-cad-suite to PATH
# This makes Verilator, cocotb, and other tools available
ENV PATH="/opt/oss-cad-suite/bin:${PATH}"

# Source the oss-cad-suite environment script on shell startup
# This ensures all environment variables are set correctly
RUN echo 'source /opt/oss-cad-suite/environment' >> /etc/bash.bashrc

# Create non-root user for running tests
# Using build args allows matching host user's UID/GID
# This ensures files created in mounted volumes have correct ownership on the host
ARG USER_ID=1000
ARG GROUP_ID=1000

# Use --non-unique flag to allow GID conflicts (e.g., macOS GID 20 = staff)
# and || true to continue if group already exists
RUN groupadd --non-unique -g ${GROUP_ID} cocotb_user || true && \
  useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash cocotb_user && \
  chown -R cocotb_user:cocotb_user /opt/oss-cad-suite

# Create workspace directories
RUN mkdir -p /workspace/sv /workspace/tests /workspace/results \
  && chown -R cocotb_user:cocotb_user /workspace

# Install Python packages for gymnasium integration tests
# Must use tabbypy3 (the Python bundled with oss-cad-suite) since cocotb uses it
COPY tests/requirements.txt /tmp/requirements.txt
RUN /opt/oss-cad-suite/bin/tabbypy3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Switch to non-root user
USER cocotb_user
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
