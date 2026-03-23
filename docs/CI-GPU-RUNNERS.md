# CI GPU Runner Infrastructure

This document explains how the GPU-based integration test infrastructure works for this repo.

## Overview

Integration tests require GPU hardware to run ML model inference. GPU VMs are expensive (~$1.62/hr for 3x T4), so they auto-scale to zero when idle. The system automatically starts runners when CI jobs need them and stops them after 15 minutes of inactivity.

## Architecture

```
GitHub webhook (workflow_job.queued)
    │
    ▼
Cloud Function (github-runner-manager)
    │
    ├── Job has "gpu" label? → Start GPU runners (3x n1-standard-4 + T4)
    ├── Job has "self-hosted" label? → Start CPU runners
    └── Neither? → Ignore

Cloud Scheduler (every 15 min)
    │
    ▼
Cloud Function (?action=check_idle)
    │
    └── No pending jobs + runner idle > 15 min? → Stop runner
```

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Cloud Function | `karaoke-gen/infrastructure/functions/runner_manager/main.py` | Starts/stops runner VMs based on demand |
| Pulumi module | `karaoke-gen/infrastructure/modules/runner_manager.py` | Deploys the function, scheduler, and IAM |
| GPU VM definitions | `karaoke-gen/infrastructure/compute/github_runners.py` | 3x n1-standard-4 with T4 GPU |
| GPU startup script | `karaoke-gen/infrastructure/compute/startup_scripts/github_runner_gpu.sh` | Installs NVIDIA drivers, Python, registers runner |
| Config | `karaoke-gen/infrastructure/config.py` | Runner count, labels, idle timeout |
| GitHub webhook | Org-level (`nomadkaraoke`) | Sends `workflow_job` events to Cloud Function |

### GPU Runner VMs

- **Count**: 3 (configurable via `NUM_GPU_RUNNERS` in config.py)
- **Machine type**: n1-standard-4 (4 vCPU, 15GB RAM) + 1x NVIDIA T4
- **Zone**: us-central1-a
- **Labels**: `self-hosted, linux, x64, gcp, gpu`
- **Startup time**: ~15-20 min (NVIDIA driver install, Python build, model download)
- **Model cache**: ~14GB of ML models pre-downloaded to `/opt/audio-separator-models/`

### Required GitHub Branch Protection Checks

The `Protect main` ruleset (ID: 529535) requires these checks to pass before merge:

- `unit-tests` — from `run-unit-tests.yaml` (runs on GitHub-hosted runners)
- `ensemble-presets` — from `run-integration-tests.yaml` (runs on GPU runners)
- `core-models` — from `run-integration-tests.yaml` (runs on GPU runners)
- `stems-and-quality` — from `run-integration-tests.yaml` (runs on GPU runners)

**IMPORTANT**: If integration test job names change (e.g., splitting or renaming jobs), you MUST update the ruleset to match. The ruleset is configured at:
https://github.com/nomadkaraoke/python-audio-separator/settings/rules/529535

To update via API:
```bash
gh api repos/nomadkaraoke/python-audio-separator/rulesets/529535 \
  --method PUT --input - <<'EOF'
{
  "name": "Protect main",
  "enforcement": "active",
  "target": "branch",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "rules": [
    {"type": "deletion"},
    {"type": "pull_request", "parameters": {
      "required_approving_review_count": 0,
      "allowed_merge_methods": ["squash"]
    }},
    {"type": "required_status_checks", "parameters": {
      "required_status_checks": [
        {"context": "unit-tests", "integration_id": 15368},
        {"context": "JOB_NAME_HERE", "integration_id": 15368}
      ]
    }}
  ]
}
EOF
```

## Troubleshooting

### Integration tests stuck in "queued"

**Symptoms**: PR checks show `pending` for `ensemble-presets`, `core-models`, `stems-and-quality`.

**Diagnosis steps**:

1. Check if GPU runners are online:
   ```bash
   gh api orgs/nomadkaraoke/actions/runners \
     --jq '.runners[] | select(.labels[].name == "gpu") | {name, status, busy}'
   ```

2. Check if GPU VMs exist:
   ```bash
   gcloud compute instances list --project=nomadkaraoke --filter="name~gpu"
   ```

3. Check Cloud Function logs for webhook delivery:
   ```bash
   gcloud logging read 'resource.labels.service_name="github-runner-manager"' \
     --project=nomadkaraoke --limit=20 \
     --format="value(timestamp,textPayload,jsonPayload.message)"
   ```

4. Check GPU runner startup logs (if VMs are RUNNING but GitHub shows offline):
   ```bash
   gcloud compute ssh github-gpu-runner-1 --zone=us-central1-a --project=nomadkaraoke \
     --command="tail -50 /var/log/github-runner-startup.log"
   ```

### GPU VMs don't exist

If `gcloud compute instances list` shows no GPU runners but Pulumi state thinks they exist:

```bash
# 1. Remove stale state (from karaoke-gen/infrastructure/ dir)
pulumi state delete "urn:pulumi:prod::karaoke-gen-infrastructure::gcp:compute/instance:Instance::github-gpu-runner-1" --target-dependents --yes
pulumi state delete "urn:pulumi:prod::karaoke-gen-infrastructure::gcp:compute/instance:Instance::github-gpu-runner-2" --target-dependents --yes
pulumi state delete "urn:pulumi:prod::karaoke-gen-infrastructure::gcp:compute/instance:Instance::github-gpu-runner-3" --target-dependents --yes

# 2. Recreate
pulumi up --yes

# 3. Re-import dependent resources that got removed (runner-manager function, IAM, scheduler)
# Check `pulumi preview` for what needs importing
```

### GPU runner startup fails (NVIDIA driver issues)

The startup script handles kernel header mismatches by upgrading the kernel and rebooting once. If the runner still fails:

```bash
# SSH in and check
gcloud compute ssh github-gpu-runner-1 --zone=us-central1-a --project=nomadkaraoke \
  --command="nvidia-smi; dkms status; uname -r"
```

See `karaoke-gen` memory file `project_gpu_runner_drivers.md` for known issues.

### Webhook not firing

Check the org-level webhook configuration:
```bash
gh api orgs/nomadkaraoke/hooks \
  --jq '.[] | select(.events[] == "workflow_job") | {id, active, config: {url: .config.url}}'
```

The webhook URL should point to: `https://us-central1-nomadkaraoke.cloudfunctions.net/github-runner-manager`

## Cost

| Scenario | Cost |
|----------|------|
| Per GPU runner hour | ~$0.54/hr (n1-standard-4 + T4) |
| 3 runners × 15 min CI run | ~$0.41 |
| Idle (scale to zero) | $0 |
| Typical daily cost (5 PRs) | ~$2 |
