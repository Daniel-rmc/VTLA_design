# VTLA Training Status

The old four-GPU run failed during DDP initialization because GPU 0 was already occupied by another long-running workload. The corrected launcher selects or validates physical GPUs before starting.

Use the following command for the current live state:

```bash
./check_training.sh
```

Runs are self-contained under `runs/<stage>/<run-name>/`:

- `config.json`: exact arguments, command, Git commit/dirty state, CUDA/PyTorch versions, physical GPU mapping, data statistics, and parameter counts.
- `launch_command.sh`: exact relaunch command.
- `train.log`: complete stdout/stderr.
- `metrics.jsonl`: global DDP epoch metrics and peak allocated GPU memory.
- `checkpoints/`: model, optimizer, scheduler, normalization statistics, and embedded run configuration.
- `exit_code`: written when the background process exits.

The current official-data Stage 2 launcher defaults to physical GPUs selected by free memory, batch size 64/GPU, BF16, both available cameras, both tactile sensors, native 8D control, a stratified 90/10 episode split, and 150 epochs. To force the last three GPUs explicitly:

```bash
./start_training_multigpu.sh stage2 3 1,2,3
```
