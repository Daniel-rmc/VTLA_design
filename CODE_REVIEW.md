# VTLA Code Review

Review date: 2026-08-09

## Resolved correctness issues

- The multi-GPU launcher always exposed GPUs `0,1,2,3`, even when GPU 0 was occupied. It now validates free memory and accepts an explicit physical GPU list.
- The fusion Transformer position tensor hard-coded batch size 1. Batches larger than 1 now receive a correctly sized tensor.
- Stage 2 previously saw only five samples (one first frame per episode). It now creates 271 causal timestep samples and predicts the future joint-position chunk.
- Missing camera/tactile datasets and state-dimension mismatches previously fell back silently or failed later. They now fail at dataset construction with the exact file/path mismatch.
- Images and joint values were not normalized. Inputs now use ImageNet normalization; joint mean/std are saved in every run config and checkpoint.
- Stage 1 used only ten frames per episode from only the left sensor, and its marker target shape did not match the 63-marker decoder. It now uses all 552 left/right frames and the expected normalized marker target.
- Stage 1 bypassed DDP by calling `model.module.compute_loss`; it now performs the loss-bearing forward through DDP.
- Stage checkpoint loading happened only on rank 0 after DDP construction. Every rank now loads identical state.
- The tactile projection layer was never used and was dimensionally wrong for configurable latent sizes. It is now applied to tactile features.
- Visual position features were discarded and tactile tokens had no spatial/source identity. Visual positions and learned camera IDs are retained; tactile tokens receive 2-D positions and sensor IDs.
- Stage 2 left contact/scale modules trainable while disabling their forward path. The complete refinement branch is now frozen.
- The adaptive refinement path ignored `refine_scale`; the learned scale is now bounded by the configured value.
- Padding predictions had no loss, and action L1 was diluted by padded elements. Padding BCE is now trained and L1 is normalized only over valid action elements.
- Stage 3 used fabricated labels equating the second half of every trajectory with contact. That invalid supervision was removed, and frozen BatchNorm/dropout modules remain in evaluation mode.
- The refinement head contained a second gate that was always bypassed by the external contact detector, leaving parameters permanently unused. The redundant gate was removed from the dual-path model.
- The single-GPU CLI registered `--lr_backbone` twice and could not start. Its argument parser and launcher now pass validation.

## Validation evidence

- Eleven tests cover timestep sampling, both UniVTAC tactile-key layouts, raw-9D to native-8D projection, deterministic stratified episode splitting, Stage 1 marker shapes, deployment tensor ownership, 2-D positions, evaluation aggregation, and refinement scaling.
- A complete one-epoch DDP smoke run succeeded on physical GPUs `1,2,3` with two cameras, two tactile sensors, batch size 32/GPU, and effective batch size 96.
- The smoke run used 9.14 GiB peak allocated memory per L40S.
- One-epoch Stage 1 and Stage 3 regression runs also completed, including checkpoint transfer and the frozen-backbone path.
- A new official-data smoke run succeeded with 100 published episodes (90 train/10 validation), native 8D control, batch size 64/GPU, effective batch 192, and BF16. It used 12.46 GiB peak memory per L40S and completed both validation and strict checkpoint deployment.
- NCCL 2.21.5 P2P collectives currently hang on this host. A minimal three-GPU health check identified the failing path; `NCCL_P2P_DISABLE=1` completes correctly and is now recorded by the launcher.

## Remaining design limitations

- The implementation has no language input, tokenizer, or language encoder. It is currently a vision-tactile-action (VTA) policy despite the VTLA name.
- The source HDF5 files do not contain a dedicated action field. The loader explicitly uses the next joint positions as an action proxy; a production imitation-learning run should store commanded actions.
- The historical `20260809_014410` run used only five locally collected demonstration episodes (271 timestep samples), so its 8% simulator result is an infrastructure baseline rather than a representative official-data result. The current pipeline targets all 100 published `grasp_classify/clean` episodes and uses a deterministic episode-level 90/10 train/validation split.
- Published UniVTAC observations contain nine joint values, while its policy API consumes eight commands. The current pipeline explicitly selects raw columns `0..7` and trains/deploys a native 8D policy; 9D conversion remains only for reproducing historical checkpoints.
- The dataset has no trustworthy contact labels. Stage 3 can receive gradients through the residual action objective, but contact detection cannot be quantitatively supervised or evaluated yet.
- The current task still has no explicit rough/plain auxiliary classification objective; classification is learned only through conditional action imitation.

Each run is stored under `runs/<stage>/<timestamp>_<git>_<gpus>/` with `config.json`, `launch_command.sh`, `train.log`, `metrics.jsonl`, checkpoints, and an eventual `exit_code`.
