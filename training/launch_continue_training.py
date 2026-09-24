#!/usr/bin/env python3
"""Launch a short, time-boxed continue-training run on AWS GPUs.

Unlike launch_universal_training.py (which builds a fresh 200k+-row corpus and
trains a randomly-initialised scoring head for 3 epochs, taking hours), this
launcher:
  1. Continues from an EXISTING option_marker.pt (full encoder+head state),
     via train_option_marker.py's --init_checkpoint, so it does not throw away
     everything the shipped model already knows.
  2. Trains on a small, pre-built local train/val jsonl pair (uploaded to S3
     once by this script) instead of rebuilding a corpus on the instance.
  3. Caps total micro-batches via --max_steps so a checkpoint is guaranteed to
     save within the time budget regardless of measured GPU throughput, and
     wraps torchrun in a wall-clock `timeout` as a last-resort backstop.

Reuses the dpkg-lock-race fix, isolated awscli install, shutdown trap, and
240-minute watchdog already validated in launch_universal_training.py.
"""

import json
import os
import shutil
import subprocess
import time

VENV_AWS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "aws"))
AWS_CLI = VENV_AWS if os.path.exists(VENV_AWS) else (shutil.which("aws") or "aws")
REGION = "us-west-2"
SUBNETS = [
    ("subnet-083ba040", "us-west-2a"),
    ("subnet-070e7461", "us-west-2b"),
    ("subnet-b75369ec", "us-west-2c"),
    ("subnet-a663228e", "us-west-2d"),
]
AMI_ID = "ami-0e24e0019a12c5b13"
CANDIDATE_TYPES = [
    ("g5.12xlarge", "4x NVIDIA A10G 96GB (On-Demand ~$5.67/hr)"),
    ("g4dn.12xlarge", "4x NVIDIA T4 64GB (On-Demand ~$3.91/hr)"),
    ("g5.4xlarge", "1x NVIDIA A10G 24GB (On-Demand ~$1.62/hr)"),
]
IAM_PROFILE = "AmazonSSMRoleForInstancesQuickSetup"

USER_DATA_TEMPLATE = """#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

cleanup() {{
  rc=$?
  echo "=== [EXIT rc=$rc] uploading log and shutting down ==="
  aws s3 cp /var/log/user-data.log {s3_target}/run.log || true
  shutdown -h now
}}
trap cleanup EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
shutdown -c 2>/dev/null || true
shutdown -h +60 &

echo "=== [CONTINUE-TRAINING START] ==="
export DEBIAN_FRONTEND=noninteractive

systemctl stop unattended-upgrades.service 2>/dev/null || true
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
for i in $(seq 1 60); do
  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  echo "waiting for dpkg lock ($i)..."; sleep 5
done
APT_OPTS="-o DPkg::Lock::Timeout=600 -y"
apt-get $APT_OPTS update && apt-get $APT_OPTS install awscli curl git

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

mkdir -p /opt/von
aws s3 cp s3://model-weight/von-marker-src.tar.gz /tmp/von-marker-src.tar.gz
tar -xzf /tmp/von-marker-src.tar.gz -C /opt/von
cd /opt/von

/root/.local/bin/uv venv --clear /opt/von/.venv
/root/.local/bin/uv pip install --python /opt/von/.venv torch torchvision transformers datasets scipy sentencepiece tiktoken accelerate pydantic awscli
export PYTHONPATH="/opt/von/src:$PYTHONPATH"

mkdir -p /opt/von/init_ckpt /opt/von/run_data
aws s3 sync {init_ckpt_s3}/ /opt/von/init_ckpt/
aws s3 cp {train_s3} /opt/von/run_data/train.jsonl
aws s3 cp {val_s3} /opt/von/run_data/val.jsonl

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs. Continue-training for up to {max_steps} steps (hard-capped)..."

timeout {timeout_s} /opt/von/.venv/bin/torchrun --nproc_per_node=$NUM_GPUS training/train_option_marker.py \\
    --train_data /opt/von/run_data/train.jsonl \\
    --val_data /opt/von/run_data/val.jsonl \\
    --base_model_id /opt/von/init_ckpt \\
    --init_checkpoint /opt/von/init_ckpt \\
    --epochs {epochs} \\
    --batch_size 8 \\
    --grad_accum_steps 2 \\
    --lr {lr} \\
    {independent_options_flag} \\
    {digit_split_flag} \\
    --max_position_embeddings 8192 \\
    --max_steps {max_steps} \\
    --s3_target {s3_target} \\
    --output_dir /opt/von/checkpoints/continue-run

aws s3 cp /var/log/user-data.log {s3_target}/run.log || true
echo "=== [CONTINUE-TRAINING COMPLETE - TERMINATING] ==="
shutdown -h now
"""


def run_aws(cmd: list) -> dict:
    full_cmd = [AWS_CLI] + cmd + ["--region", REGION, "--output", "json"]
    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"AWS CLI error: {res.stderr.strip()}")
    if not res.stdout.strip():
        return {}
    return json.loads(res.stdout)


def launch(
    train_jsonl: str,
    val_jsonl: str,
    init_ckpt_dir: str,
    s3_target: str,
    init_ckpt_s3: str = "s3://model-weight/von-continue-init-ckpt",
    max_steps: int = 6000,
    lr: float = 1e-5,
    epochs: int = 1,
    independent_options: bool = False,
    digit_split: bool = False,
    timeout_s: int = 2100,
    on_demand: bool = True,
):
    print("================================================================")
    print("  VON CONTINUE-TRAINING LAUNCHER")
    print(f"  max_steps:  {max_steps}  (hard cap, saves checkpoint on trigger)")
    print(f"  lr:         {lr}")
    print(f"  timeout:    {timeout_s}s wall-clock backstop around torchrun")
    print(f"  S3 target:  {s3_target}")
    print("================================================================\n")

    print(f"Uploading init checkpoint {init_ckpt_dir} -> {init_ckpt_s3} ...")
    subprocess.run([AWS_CLI, "s3", "sync", init_ckpt_dir, init_ckpt_s3 + "/",
                    "--region", REGION], check=True)
    train_s3 = init_ckpt_s3.rsplit("/", 1)[0] + "/continue_train.jsonl"
    val_s3 = init_ckpt_s3.rsplit("/", 1)[0] + "/continue_val.jsonl"
    subprocess.run([AWS_CLI, "s3", "cp", train_jsonl, train_s3, "--region", REGION], check=True)
    subprocess.run([AWS_CLI, "s3", "cp", val_jsonl, val_s3, "--region", REGION], check=True)
    print("Upload complete.\n")

    user_data_path = "/tmp/user_data_continue.sh"
    with open(user_data_path, "w") as f:
        f.write(USER_DATA_TEMPLATE.format(
            s3_target=s3_target,
            init_ckpt_s3=init_ckpt_s3,
            train_s3=train_s3,
            val_s3=val_s3,
            max_steps=max_steps,
            lr=lr,
            epochs=epochs,
            independent_options_flag="--independent_options" if independent_options else "",
            digit_split_flag="--digit_split" if digit_split else "",
            timeout_s=timeout_s,
        ))

    instance_id = None
    for itype, desc in CANDIDATE_TYPES:
        print(f"\nEvaluating instance type: {itype} [{desc}]...")
        for subnet_id, az in SUBNETS:
            print(f"  -> Trying {itype} in {az} ({subnet_id})...")
            try:
                run_args = [
                    "ec2", "run-instances",
                    "--image-id", AMI_ID,
                    "--instance-type", itype,
                    "--subnet-id", subnet_id,
                    "--iam-instance-profile", f"Name={IAM_PROFILE}",
                    "--user-data", f"file://{user_data_path}",
                    "--count", "1",
                    "--tag-specifications", json.dumps([{
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": "von-continue-training"}]
                    }]),
                ]
                if not on_demand:
                    run_args.extend(["--instance-market-options", json.dumps({"MarketType": "spot"})])
                res = run_aws(run_args)
                instance_id = res["Instances"][0]["InstanceId"]
                print(f"\n-> SUCCESS! Launched {itype} in {az}: {instance_id}")
                break
            except Exception as e:
                err = str(e)
                if "InsufficientInstanceCapacity" in err or "Unsupported" in err or "SpotMaxPriceTooLow" in err:
                    print(f"     Capacity unavailable in {az}.")
                    continue
                print(f"     Failed: {err}")
        if instance_id:
            break

    if not instance_id:
        print("\nAll candidate pools exhausted.")
        return None

    print("\nWaiting for instance to enter 'running' state...")
    while True:
        desc = run_aws(["ec2", "describe-instances", "--instance-ids", instance_id])
        state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"  -> Instance {instance_id} status: {state}")
        if state == "running":
            break
        time.sleep(5)

    print(f"\nContinue-training instance is RUNNING: {instance_id}")
    print(f"Artifacts will upload to {s3_target} on completion or max_steps trigger.")
    return instance_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch a short Von continue-training run.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--init-ckpt", required=True)
    parser.add_argument("--s3-target", required=True)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--independent-options", action="store_true")
    parser.add_argument("--digit-split", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=2100)
    args = parser.parse_args()

    launch(
        train_jsonl=args.train,
        val_jsonl=args.val,
        init_ckpt_dir=args.init_ckpt,
        s3_target=args.s3_target,
        max_steps=args.max_steps,
        lr=args.lr,
        epochs=args.epochs,
        independent_options=args.independent_options,
        digit_split=args.digit_split,
        timeout_s=args.timeout_s,
    )
