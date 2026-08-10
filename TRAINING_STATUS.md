# VTLA Training Status

## Latest completed official-data run

The native-8D Stage 2 run completed successfully on physical GPUs `1,2,3`:

- Run directory: `runs/stage2/20260809_155942_1073ae9_gpu123`
- Dataset: ModelScope `byml2024/UniVTAC/grasp_classify/clean`, all 100 trajectories
- Split: 90 train / 10 validation trajectories, stratified and fixed by seed `20260809`
- Training: 150 epochs, BF16, batch 64/GPU (effective 192), exit code 0
- Selected checkpoint: epoch 130, chosen by held-out deployment first-step MAE
- UniVTAC evaluation: 80/100 success on seeds `1000000..1000099`

The full report is preserved at `runs/stage2/20260809_155942_1073ae9_gpu123/eval/EVALUATION_SUMMARY.md`.

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

The official-data Stage 2 launcher defaults to physical GPUs selected by free memory, batch size 64/GPU, BF16, both available cameras, both tactile sensors, native 8D control, a stratified 90/10 episode split, and 150 epochs. To force the last three GPUs explicitly:

```bash
./start_training_multigpu.sh stage2 3 1,2,3
```
