# Reproducing the paper

Complete [installation](INSTALL.md) first. This release evaluates ACT only and includes the scripted expert, replay, random and hold calibration policies.

## Task and training recipe

The Franka Panda lifts a 5 cm cube. Demonstrations cover radii {0.35, 0.40, 0.45, 0.50} m and bearings {−30, −20, −10, 0, 10, 20, 30} degrees. The envelope is `r ∈ [0.325, 0.525] m, |θ| ≤ 35°`. Evaluation includes 28 training positions, 21 withheld interior positions, 35 radial test positions and 24 angular test positions, with two episodes each. The test bands extend 5–25 cm radially and 10–30 degrees angularly beyond the envelope.

Each collection has 15 demonstrations per training position, for 420 total. Arm start jitter is Gaussian with σ=0.02 rad per joint. ACT uses a fixed random 336/84 training/validation partition, ImageNet-pretrained ResNet-18, chunk length 16, batch size 8, learning rate 1e-5, KL weight 10 and 2000 epochs. The lowest validation-loss checkpoint is evaluated. Statistics cover all 420 end-padded demonstrations, including validation, with a standard-deviation floor of 0.01. Predictions are not clipped. Joint PD stiffness/damping are 400/80; EEF commands use one damped-least-squares IK step with λ=0.01.

## Camera collections

Collection A supplies the interface and anchoring comparisons and head-only zeroed-state control. Its wrist lens is inside the hand and produces a near-uniform dark image; set `ESCAPE_WRIST_LEGACY=1` for both collection and evaluation. Collection B supplies all camera comparisons and the wrist-only zeroed-state control, with the wrist camera beside the fingers; set `ESCAPE_WRIST_LEGACY=0`. The reproduction driver selects the matching mount. Head images are 672×376 and wrist images 424×240. Images are halved; when both are used, the wrist image is first resized to match the head image.

## Stage metrics

Let E be the demonstrated envelope, o the object's nominal spawn position, and p the grasp point between the fingertips, projected onto the table plane.

- **Escape:** `max_t max(b(p_t), 0)`, in metres, where `b` is the signed boundary excursion defined in [spec.py](../escape_ood/suites/spec.py). The count threshold is strictly greater than 1 cm. Angular excursion uses arc length; corner excursion uses distance to the nearest corner.
- **Approach:** `clip(1 − min_t ||p_t − o|| / d(o,E), 0, 1)`, with Euclidean distance `d` to the envelope. Counts use ≥0.25. This measures progress toward the object, rather than departure in any direction.
- **Strict completion:** the cube centre remains at least 1 cm above its rest height and within 6 cm of the grasp point in the plane at three consecutive checks, 15 control steps apart. The control rate is 30 Hz and each rollout lasts at most 600 steps. For older records without a sustained-check field, the scorer conservatively requires a successful early exit.

Stages are scored independently. Of 495 strict test completions in the main table, 494 satisfy both stage thresholds; the exception has approach 0.23 and moved the cube. The summary records that exception explicitly. Contact and support are not measured.

## Full reproduction, in order

Run commands from the repository root. Keep generated datasets and checkpoints under `data/output/`. Use a fresh output directory for a fresh collection; the collector appends to existing archives.

1. **Build the table and cube assets, then check achievability.** In a terminal with the simulation environment activated:

   ```bash
   python3 scripts/reproduce.py scene --execute
   python3 scripts/reproduce.py gate --execute
   ```

   The exhaustive gate runs every cell, with two starts per cell. The checked-in per-cell gate record is used by the specification test; a new gate writes under `data/output/gate`.

2. **Collect and verify both camera collections.**

   ```bash
   python3 scripts/reproduce.py collect --execute
   python3 scripts/reproduce.py verify --execute
   ```

   The driver preserves the collection partitioning: four archives for A, three for B. This also preserves conversion ordering. It runs partitions sequentially; each process owns one archive. Verification checks coverage, episode counts, commanded/achieved EEF arrays, rate limits and phase spans. The dark wrist observation in A is expected.

3. **Convert the training split.**

   ```bash
   python3 scripts/reproduce.py convert --execute
   ```

   This creates each required ACT dataset, including the rotation, anchoring and zeroed-state controls. Query-anchored actions are computed by the training loader from absolute commands and measured anchors. Other residuals are computed during conversion. Only the training split enters these datasets.

4. **Train.** Train one named run, or omit `--run` to run all reported recipes sequentially:

   ```bash
   python3 scripts/reproduce.py train --run act --execute
   ```

   Repeat for every name printed by `list`. There are three seeds for every configuration. Checkpoints and their `dataset_stats.pkl` are saved together under `data/output/train/<run>/lift/`.

5. **Serve and evaluate each checkpoint.** In an ACT terminal:

   ```bash
   python3 scripts/reproduce.py serve --run act --port 6000 --execute
   ```

   In a separate terminal, source the simulation environment and run:

   ```bash
   python3 scripts/reproduce.py evaluate --run act --port 6000 --execute
   ```

   Evaluate the same run that is being served. The driver runs interior, test and training splits, with two episodes per cell, and writes the filenames expected by the manifest. Repeat this pair of commands for each run. Stop your server after its evaluation. Closed-loop evaluation is not bitwise repeatable; the paper reports repeatability measurements in the summary.

6. **Calibrate the evaluator with the scripted expert and recorded-action replay.**

   ```bash
   python3 scripts/reproduce.py calibrate --execute
   ```

   The expert must complete all 216 episodes. Replay uses the demonstration's own reset seed and recorded actions on training cells and needs no policy server. `eval_policy.py --policy random` and `--policy hold` provide additional floors. The calibration driver checks all reported action decoding formats that have absolute recordings.

7. **Render the inputs needed by the figures.** Render the head scene used in the teaser:

   ```bash
   venvs/sim/bin/python escape_ood/sim/tools/render_scene.py --suite lift --cell 0.575,0
   ```

   The teaser also reads the wrist crop from collection B. For the rollout overlay, serve each of `act`, `act_eef_abs_wo` and `act_joint_chunk` in turn, then use the simulation terminal:

   ```bash
   python3 scripts/reproduce.py fig3 --run act --port 6000 --execute
   ```

   Substitute the matching run each time. This re-evaluates the shared figure position with a side camera and depth; policies receive only their training cameras. These rendered rollouts are separate from the numerical evaluation runs.

8. **Recompute the table, diagnostics and figures from raw JSONs.**

   ```bash
   python3 scripts/reproduce.py report --execute
   ```

   The report stage writes `outputs/paper_cells.json`, `outputs/figures/TABLE_cells.tex` and the three paper figures. The summary includes radial/angular stages, band medians, fit/interpolation controls, envelope sensitivity, strict/native scoring, 3D proximity, moved-cube and height checks, and stage-order violations. Optional repeatability checks require the additional evaluations described below.

Raw evaluation JSONs, demonstrations, rendered rollout sequences and checkpoints are **not included**. The small published count summary is included. Set `ESCAPE_RESULTS_DIR` to an alternative result root containing `eval/`, `collect/`, `collect_wrist/` and `media/`. Set `ESCAPE_FIGURES_DIR` to change figure output location; `ESCAPE_MEDIA_DIR` can separately select the overlay rollout directory. Table and approach-figure regeneration only need evaluation JSONs. The teaser and overlay also require images and depth, which cannot be recovered from path JSONs.

To reproduce the repeatability diagnostic, serve each rotation/query-anchor control and run `python3 scripts/reproduce.py repeatability --run <run> --execute` in the simulation terminal. This evaluates the alternate mount using the run name as its tag, plus an unchanged-setting `_rep` evaluation for the two seed-zero query-anchor controls. These repeat runs do not train additional models. Then add `--repeatability` to `scripts/paper/summarize.py --diagnostics`.

## Measured resources

The ACT batch-size measurement used an A6000 with 48 GiB. Training and evaluation times below are ranges from the released configurations' recorded logs; shared GPU load and camera input affected them.

| Operation | Measured time | GPU memory |
|---|---|---|
| ACT training, batch 8, 2000 epochs | 1:09:50–3:32:08 per run | 1.56 GiB representative training-step peak |
| Test evaluation, 118 episodes | 0:58:08–6:06:28 summed episode time | Simulator and policy-server memory were not separately tabulated |

The training memory measurement excludes Isaac Sim, the policy server and allocator overhead. Keep the paper's batch size when reproducing its recipe. Scene startup, conversion and collection timings were not tabulated for this release. [results/runtime_measurements.json](../results/runtime_measurements.json) retains run names and measured durations.
