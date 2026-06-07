# BridgeEngine Runtime Storage On D:

Last updated: 2026-06-07

## Current Situation

`D:` has plenty of space and already holds the genuinely large research inputs:

```text
D:\bridgedata_v2_subset
D:\lewm_runs
D:\hf_cache
D:\extraction_models
```

`C:` is the pressure point. The repo itself should keep code and small tracked artifacts on `C:`, but ignored runtime data should live on `D:`.

Measured on 2026-06-07:

```text
C: free space: about 19 GB after deleting the failed repo-local head-to-head run
D: free space: about 1.4 TB
annotation_pipeline\.venv: about 5.36 GB
annotation_pipeline\bridgeengine_data: about 0.13 GB
LeWM_testbed\.venv: about 5.71 GB
LeWM_testbed\outputs: about 1.92 GB
```

## What To Move

Safe to move behind NTFS junctions:

```text
annotation_pipeline\bridgeengine_data
annotation_pipeline\training_cuts
annotation_pipeline\.venv                  optional
LeWM_testbed\.venv                         optional
LeWM_testbed\outputs                       optional
```

Do not move these tracked artifact directories wholesale:

```text
bench_results
scale_results
head_to_head_results\preregistered_100
figures
gold_sets
```

Those are small and include committed files. The full head-to-head runner already writes bulky outputs to:

```text
D:\lewm_runs\bridgeengine_head_to_head\run_100
```

## Recommended Migration

Dry run first:

```powershell
.\scripts\migrate_runtime_to_d.ps1 -IncludeVenv -IncludeLewmTestbed
```

Apply:

```powershell
.\scripts\migrate_runtime_to_d.ps1 -Apply -IncludeVenv -IncludeLewmTestbed
```

This moves the bytes to:

```text
D:\bridgeengine_runtime\annotation_pipeline
D:\bridgeengine_runtime\LeWM_testbed
```

and leaves junctions at the original paths. Existing commands keep working because paths like:

```text
C:\Users\Kevin\projects\annotation_pipeline\.venv
C:\Users\Kevin\projects\annotation_pipeline\bridgeengine_data
```

still exist as junctions.

## Validation

After applying:

```powershell
Get-Item bridgeengine_data,training_cuts,.venv -Force | Select-Object Name,LinkType,Target
.\.venv\Scripts\python.exe -m pytest tests\test_head_to_head.py -q
.\scripts\run_head_to_head_100.ps1 -PrepareOnly
```

If those pass, the repo is still wired correctly while bulky runtime bytes live on `D:`.

## Environment Variable Alternative

BridgeEngine also supports:

```powershell
$env:BRIDGEENGINE_DATA_ROOT = "D:\bridgeengine_runtime\annotation_pipeline\bridgeengine_data"
```

The junction approach is still preferred on this machine because it protects older commands and notebooks that assume repo-relative paths.
