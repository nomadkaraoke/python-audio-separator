# CI GPU Runner Infrastructure

This document explains how the GPU-based integration test infrastructure works for this repo.

## Overview

Integration tests require GPU hardware to run ML model inference. GPU VMs are
expensive, so the fleet is **fully ephemeral**: every queued CI job gets a
fresh single-use GCE VM created on demand from a pre-baked image, and the VM
deletes itself when the job finishes. Nothing runs (or costs money) while CI
is idle.

> **History**: until 2026-05-18 this was a fixed pool of long-lived VMs
> (`github-gpu-runner-{1,2,3}`) that were started/stopped on demand. Phase 4
> of the ephemeral-runners rollout (karaoke-gen PR #780) deleted that pool.
> If you see references to starting/stopping runner VMs, they're stale.

## Architecture

```
GitHub org webhook (workflow_job.queued)
    │
    ▼
Cloud Function (github-runner-manager)
    │
    ├── resolve image family from job labels:
    │     gpu          → gha-runner-gpu         (Linux, n1-standard-4 + T4)
    │     docker-build → gha-runner-build       (Linux, e2-standard-8)
    │     otherwise    → gha-runner-general     (Linux, e2-standard-4)
    │     windows+gpu  → gha-runner-gpu-windows (Windows Server + T4; in progress)
    │
    ├── mint a JIT (just-in-time) ephemeral runner config via GitHub API
    └── create a single-use GCE VM from the family image
            │
            ▼
        VM boots, runs ONE job (`run.sh --jitconfig`), de-registers,
        shuts itself down (boot disk auto-deletes on VM delete)

Cloud Scheduler (every 15 min)
    │
    ▼
Cloud Function (?action=cleanup_orphans)
    │
    ├── delete VMs whose runner is gone (age > 30 min) or hung (age > 120 min)
    └── de-register zombie runner registrations with no live VM
```

### Components (all in the karaoke-gen repo)

| Component | Location | Purpose |
|-----------|----------|---------|
| Dispatcher + cleanup | `karaoke-gen/infrastructure/functions/runner_manager/ephemeral.py` | Family resolution, JIT config, VM create, orphan cleanup |
| Webhook entry point | `karaoke-gen/infrastructure/functions/runner_manager/main.py` | Signature verification, event routing |
| Image bake workflow | `karaoke-gen/.github/workflows/build-runner-images.yml` | Builds `gha-runner-<variant>` image families (monthly cron + manual dispatch) |
| Image provisioning | `karaoke-gen/infrastructure/scripts/runner-image-provision.sh` | Installs NVIDIA driver, Python, Poetry, runner binary, ~14GB model cache |
| GitHub webhook | Org-level (`nomadkaraoke`) | Sends `workflow_job` events to the Cloud Function |

### GPU runner VMs (ephemeral)

- **Name pattern**: `gha-gpu-<hex>` (also `gha-general-<hex>`, `gha-build-<hex>`)
- **Machine type**: n1-standard-4 (4 vCPU, 15GB RAM) + 1× NVIDIA T4
- **Zones**: us-central1-a primary, us-east4-c fallback on stockout
- **Image**: `gha-runner-gpu` family — NVIDIA driver and ~14GB of models at
  `/opt/audio-separator-models` are baked in, so job start is fast
- **Lifetime**: one CI job (~7 min typical), then self-destructs

### Runner labels and `runs-on`

Linux GPU runners advertise: `self-hosted, linux, x64, gcp, gpu`.
Windows GPU runners (in progress) advertise: `self-hosted, windows, x64, gcp, gpu`.

**Always include the OS label in `runs-on`** (e.g.
`[self-hosted, linux, gpu]`). GitHub schedules a job onto any runner whose
labels are a superset of the job's — a bare `[self-hosted, gpu]` job could be
picked up by a Windows GPU runner.

## Windows coverage

Two tiers (added 2026-07 for RoFormer/DirectML support, issue #292):

1. **`windows-cpu-integration`** — GitHub-hosted `windows-latest` (free),
   runs one model per DirectML-relevant architecture (RoFormer, MDX, VR)
   end-to-end on CPU. Models are cached via `actions/cache`. Python 3.12
   (torch-directml has no 3.13 wheels, and the DML jobs must match).
2. **`windows-directml`** (planned) — self-hosted ephemeral Windows Server +
   T4 VM (`gha-runner-gpu-windows` family), runs separation with
   `--use_directml` and compares output quality against CPU results. The
   image uses the NVIDIA **GRID** driver (WDDM mode) — the datacenter driver
   puts the T4 in TCC mode, which has no DirectX support and breaks DirectML.

## Required GitHub branch protection checks

The `Protect main` ruleset (ID: 529535) requires these checks to pass before merge:

- `unit-tests` — from `run-unit-tests.yaml` (GitHub-hosted runners)
- `ensemble-presets` — from `run-integration-tests.yaml` (GPU runners)
- `core-models` — from `run-integration-tests.yaml` (GPU runners)
- `stems-and-quality` — from `run-integration-tests.yaml` (GPU runners)

`windows-cpu-integration` and `windows-directml` are intentionally **not**
required yet; they get added to the ruleset (and to the gate job's failure
conditions) after a stabilization period.

**IMPORTANT**: If integration test job names change (e.g., splitting or
renaming jobs), you MUST update the ruleset to match. The ruleset is
configured at:
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

1. Check the Cloud Function's response to the webhook — this is the most
   common failure point. A 503 with `{"error": "HTTP Error 403"}` means the
   `github-runner-pat` secret is expired/unauthorized (it's used to mint JIT
   configs). Check recent deliveries on the org webhook:
   ```bash
   gh api orgs/nomadkaraoke/hooks   # find the workflow_job webhook id
   gh api orgs/nomadkaraoke/hooks/<id>/deliveries --paginate | head -50
   ```
   Fix: rotate the PAT and add a new secret version:
   ```bash
   echo -n "<new-pat>" | gcloud secrets versions add github-runner-pat \
     --project=nomadkaraoke --data-file=-
   ```

2. Check whether ephemeral VMs were actually created:
   ```bash
   gcloud compute instances list --project=nomadkaraoke \
     --filter='labels.purpose="gha-ephemeral-runner"'
   ```

3. Check Cloud Function logs for dispatch errors:
   ```bash
   gcloud logging read 'resource.labels.service_name="github-runner-manager"' \
     --project=nomadkaraoke --limit=20 \
     --format="value(timestamp,textPayload,jsonPayload.message)"
   ```

4. Check runner registrations GitHub-side:
   ```bash
   gh api orgs/nomadkaraoke/actions/runners \
     --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
   ```

### VM boots but the job never starts

The orphan-cleanup pass logs the VM's serial console before deleting a VM
that never registered — check the Cloud Function logs (above). To look at a
live VM yourself:

```bash
gcloud compute instances get-serial-port-output <vm-name> \
  --zone=us-central1-a --project=nomadkaraoke --port=1 | tail -100
```

### Image problems (driver failures, missing models)

Images are rebuilt monthly (and on demand) by the `Build GHA Runner Images`
workflow in karaoke-gen. To rebuild just the GPU image:

```bash
gh workflow run build-runner-images.yml --repo nomadkaraoke/karaoke-gen \
  -f variants=gpu
```

See karaoke-gen memory/docs for known NVIDIA driver issues
(`project_gpu_runner_drivers.md`).

## Cost

| Scenario | Cost |
|----------|------|
| Per Linux GPU VM-hour | ~$0.54/hr (n1-standard-4 + T4, on-demand) |
| 3 parallel jobs × ~10 min | ~$0.27 per CI run |
| Per Windows GPU VM-hour (planned) | ~$0.72/hr (adds Windows Server license) |
| Idle | $0 (no VMs exist between jobs) |
