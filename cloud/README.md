# Running the training on a cloud GPU

This POC trains fine locally (a ~2.3 M-param U-Net, a few minutes on an Apple
Silicon MPS or any GPU). This folder packages it so you can run the **exact same
pipeline** on a one-off cloud GPU and get the checkpoints + prediction PNGs back.

- `Dockerfile` — self-contained image (Python 3.12 + uv + the package). On Linux
  x86_64 the default PyPI torch wheel is CUDA-enabled, so the container uses the
  GPU automatically (`train.py` picks `cuda`).
- `entrypoint.sh` — runs the whole thing inside the container: **prepare data if
  missing → train → eval**, writing checkpoints to `$OUT_DIR` and prediction PNGs
  to `$OUTPUTS_DIR` (both under the `/workspace` volume by default). Knobs are env
  vars: `EPOCHS, BATCH_SIZE, EMPTY_WEIGHT, MAX_VOCAB, DATA_DIR, OUT_DIR, OUTPUTS_DIR`.
- `run_local_gpu.sh` — build + run on a local NVIDIA GPU to sanity-check first.

Because the container regenerates its own data (it politely downloads the seed
books), the only thing you must persist is the **output** directory.

> **Build note (Apple Silicon):** always build with `--platform linux/amd64`, or
> the image fails on the x86 GPU host with `exec format error`.

---

## TL;DR recommendation

| Provider | Cheapest single GPU | Effective $/job¹ | Simplicity | Pick it when |
|---|---|---|---|---|
| **GCP — Vertex AI custom job** | T4 (16 GB) ~$0.59/hr | **~$0.05–0.15** | ★★★ auto-provisions **and auto-deletes** the GPU; one command | You want the least-footgun managed path (recommended) |
| **AWS — EC2 `g4dn.xlarge` Spot** | T4 (16 GB) ~$0.28/hr spot | **~$0.02–0.05 (cheapest)** | ★★ self-terminating Spot VM; needs ECR + IAM profile + S3 + quota | You want the absolute lowest cost and don't mind more setup |
| **Nebius — L40S Compute VM** | **L40S (48 GB)** ~$0.65–0.90/hr preemptible | ~$0.40–1.00 | ★★ SSH + `docker run`; EU-only | You're already on Nebius (note: no T4/L4 — L40S is the floor, overkill here) |

¹ A few-minute run including boot + image pull. All three are **per-second / per-minute
billed**; storage (registry image, checkpoints) is the only thing that lingers.

**The #1 blocker on every provider is GPU quota** — new accounts often start at 0.
Request an increase (a single T4 / 4 GPU-vCPUs) before launching; approval is
usually minutes but can take up to ~2 days. GPUs are **not** in any always-free tier.

---

## Option A (recommended): GCP Vertex AI custom job

Auto-provisions a T4, pulls your image, runs it, and **tears the GPU down when the
process exits** — nothing to forget. Checkpoints persist to a GCS bucket that
Vertex auto-mounts at `/gcs/<bucket>`.

```bash
# 0. vars
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1                       # best T4 availability
export REPO=ml IMAGE=factorio-train TAG=v1
export BUCKET=${PROJECT_ID}-factorio-ml
export IMG_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}

# 1. one-time setup
gcloud auth login && gcloud config set project ${PROJECT_ID}
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com storage.googleapis.com
gcloud artifacts repositories create ${REPO} --repository-format=docker --location=${REGION}
gcloud storage buckets create gs://${BUCKET} --location=${REGION}
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# 2. build (amd64!) + push the image from the repo root
docker buildx build --platform linux/amd64 -f cloud/Dockerfile -t ${IMG_URI} --load .
docker push ${IMG_URI}

# 3. launch the job. Point the container's OUT/OUTPUTS/DATA dirs at the GCS mount
#    so artifacts persist; the container downloads + builds the dataset itself.
cat > vertex-job.yaml <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-4
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: ${IMG_URI}
      env:
        - name: OUT_DIR
          value: /gcs/${BUCKET}/runs/poc_cloud
        - name: OUTPUTS_DIR
          value: /gcs/${BUCKET}/outputs
        - name: DATA_DIR
          value: /tmp/data
        - name: EPOCHS
          value: "24"
EOF
gcloud ai custom-jobs create --region=${REGION} --display-name=factorio-unet --config=vertex-job.yaml

# 4. watch + retrieve (JOB_ID is in the create output: .../customJobs/JOB_ID)
gcloud ai custom-jobs stream-logs ${JOB_ID} --region=${REGION}
gcloud storage cp -r gs://${BUCKET}/runs/poc_cloud ./runs/
gcloud storage cp -r gs://${BUCKET}/outputs ./outputs/
```

Cheaper-but-manual GCP alternative: a **Compute Engine Spot T4** (~$0.14/hr) with
the Deep Learning image, `docker run --gpus all -e OUT_DIR=/workspace/runs -v
/home/out:/workspace ${IMG_URI}`, `gcloud compute scp` the results, then **delete
the VM** (`gcloud compute instances delete ...`).

## Option B (cheapest): AWS EC2 `g4dn.xlarge` Spot, self-terminating

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_URI=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/factorio-unet
export BUCKET=factorio-ckpt-$ACCOUNT_ID

# image -> ECR
aws ecr create-repository --repository-name factorio-unet --region $AWS_REGION || true
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${ECR_URI%/*}
docker buildx build --platform linux/amd64 -f cloud/Dockerfile -t $ECR_URI:latest --load .
docker push $ECR_URI:latest
aws s3 mb s3://$BUCKET --region $AWS_REGION

# user-data: pull -> run -> push to S3 -> self-terminate
cat > userdata.sh <<EOF
#!/bin/bash
set -euxo pipefail
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${ECR_URI%/*}
docker pull $ECR_URI:latest
mkdir -p /opt/out
docker run --gpus all -e OUT_DIR=/workspace/runs -e OUTPUTS_DIR=/workspace/outputs -v /opt/out:/workspace $ECR_URI:latest
aws s3 cp /opt/out s3://$BUCKET/run1/ --recursive
shutdown -h now
EOF

# launch Spot g4dn.xlarge on the Deep Learning Base GPU AMI (Docker + NVIDIA toolkit preinstalled).
# Needs a one-time IAM instance profile with ECR-read + S3 access (see cloud research notes).
AMI_ID=$(aws ec2 describe-images --region $AWS_REGION --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
aws ec2 run-instances --region $AWS_REGION --image-id $AMI_ID --instance-type g4dn.xlarge \
  --instance-market-options '{"MarketType":"spot"}' \
  --iam-instance-profile Name=unet-ec2-profile \
  --instance-initiated-shutdown-behavior terminate \
  --user-data file://userdata.sh
# retrieve after it self-terminates:
aws s3 cp s3://$BUCKET/run1/ ./outputs --recursive
```

The fully-managed AWS alternative is a **SageMaker training job** (`aws sagemaker
create-training-job ... --enable-managed-spot-training`), which auto-provisions and
tears down but requires the container to write to `/opt/ml/model` and a SageMaker
execution role.

## Option C: Nebius L40S Compute VM (EU-only; L40S is the smallest GPU)

```bash
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash && exec -l $SHELL
nebius profile create                       # browser login, pick tenant/project
export REGION=eu-north1 PROJECT_ID=project-XXXX
nebius registry create --name unet-cr --parent-id $PROJECT_ID
export CR=cr.$REGION.nebius.cloud/$(nebius registry list --parent-id $PROJECT_ID --format json | jq -r '.items[]|select(.metadata.name=="unet-cr").metadata.id' | sed 's/^registry-//')
nebius iam get-access-token | docker login cr.$REGION.nebius.cloud --username iam --password-stdin
docker buildx build --platform linux/amd64 -f cloud/Dockerfile -t $CR/unet:latest --load .
docker push $CR/unet:latest
# create a preemptible L40S VM from the ubuntu24.04-cuda12 image (see cloud research for the full
# `nebius compute instance create` invocation with SSH cloud-init + public IP), then:
ssh ubuntu@$VM_IP "echo $(nebius iam get-access-token) | sudo docker login cr.$REGION.nebius.cloud -u iam --password-stdin && \
  mkdir -p ~/out && sudo docker run --rm --gpus all -e OUT_DIR=/workspace/runs -e OUTPUTS_DIR=/workspace/outputs -v ~/out:/workspace $CR/unet:latest"
scp -r ubuntu@$VM_IP:~/out ./checkpoints
nebius compute instance delete --id $VM_ID        # and delete the leftover boot disk
```

---

## Honest take for *this* job

The model is so small it trains in minutes on a laptop GPU — **cloud GPU is not
needed for the POC itself**. The value of this setup is (a) reproducibility and
(b) a ready path to **scale up** (much larger vocab/corpus/model, longer training,
hyperparameter sweeps). For a quick one-off, **GCP Vertex** is the least likely to
leave a GPU running and bill you; **AWS g4dn Spot** is the cheapest if you're
comfortable with the extra IAM/ECR/S3 wiring. **Nebius** works but its smallest GPU
(L40S, 48 GB) is wild overkill and EU-only.

Pricing/commands were gathered from official docs on 2026-06-27; verify the live
rate in the console before launching (GPU prices and free-tier terms change).
