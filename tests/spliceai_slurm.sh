#!/bin/bash
#SBATCH --job-name=rna_fm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               # <-- spawn 2 tasks
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH -C "h100"
#SBATCH --gpus-per-task=1                 # <-- 1 GPU each task

# Host path to CA certificates (discovered via ls)
HOST_CA_CERT_PATH="/etc/ssl/certs/ca-bundle.crt"
# Container path (match the host)
CONTAINER_CA_CERT_PATH="/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"

# need to go into the correct container first!!
# The Slurm job itself runs on the host. We use apptainer exec to run the command inside the container.
# Note: The paths to the container image, overlay, and script are relative to where sbatch is run.
# Ensure the working directory is correct or use absolute paths if necessary.
apptainer exec --nv --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} --bind ./:/mnt/RNABenchmark --overlay ../pytorch_container/overlay.img ../pytorch_container/pytorch_25.03-py3.sif bash /mnt/RNABenchmark/tests/spliceai.sh