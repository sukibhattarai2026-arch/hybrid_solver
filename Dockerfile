# PyTorch with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Working directory
WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir \
    numpy \
    h5py \
    matplotlib

# Copy training code
COPY rieman_mlp_baseline.py /app/

# Run training
ENTRYPOINT ["python", "/app/rieman_mlp_baseline.py"]