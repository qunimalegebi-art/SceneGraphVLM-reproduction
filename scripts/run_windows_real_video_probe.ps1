param(
    [string]$SamjamRoot = "D:\samjam\src\SAMJAM-official\SAMJAM-main",
    [string]$Python = "D:\software\conda\envs\scenegraphvlm\python.exe"
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:IMAGE_MAX_TOKEN_NUM = "1024"

$frames = Join-Path $SamjamRoot "input\epic_move_cup_4_frames"
$samjamGraphs = Join-Path $SamjamRoot "output\ad7828ec-f8eb-4728-bc54-3621e9d8d808\scene_graph_output"

& $Python tools\prepare_real_video_probe.py `
    --frames-dir $frames `
    --output-dir experiments\real_video_probe

& $Python metrics\qwen-bench\infer\GT-prompt\infer_swift_gt_prompt.py `
    --model checkpoints\PVSG `
    --test-jsonl experiments\real_video_probe\probe.jsonl `
    --output-dir experiments\real_video_probe\results `
    --run-name scenegraphvlm-pvsg-4frames `
    --batch-size 1 `
    --infer-backend transformers `
    --max-new-tokens 512 `
    --temperature 0 `
    --torch-dtype bfloat16 `
    --force

& $Python tools\render_scenegraphvlm_results.py `
    --pred-jsonl experiments\real_video_probe\results\scenegraphvlm-pvsg-4frames.jsonl `
    --output-dir experiments\real_video_probe\rendered

& $Python tools\prepare_constrained_relation_probe.py `
    --images-dir experiments\real_video_probe\frames_640x480 `
    --samjam-graph-dir $samjamGraphs `
    --output experiments\constrained_relation_probe\probe.jsonl

$prefix = "<answer>`n"
& $Python metrics\qwen-bench\infer\GT-prompt\infer_swift_gt_prompt.py `
    --model checkpoints\PVSG `
    --test-jsonl experiments\constrained_relation_probe\probe.jsonl `
    --output-dir experiments\constrained_relation_probe\results `
    --run-name scenegraphvlm-constrained-4frames `
    --batch-size 1 `
    --infer-backend transformers `
    --max-new-tokens 256 `
    --temperature 0 `
    --torch-dtype bfloat16 `
    --response-prefix $prefix `
    --force

& $Python tools\analyze_constrained_probe.py `
    --input-jsonl experiments\constrained_relation_probe\probe.jsonl `
    --pred-jsonl experiments\constrained_relation_probe\results\scenegraphvlm-constrained-4frames.jsonl `
    --output experiments\constrained_relation_probe\summary.json

& $Python tools\make_side_by_side.py `
    --samjam-vis-dir (Join-Path $SamjamRoot "output\ad7828ec-f8eb-4728-bc54-3621e9d8d808\vis_output") `
    --scene-vis-dir experiments\real_video_probe\rendered\frames `
    --output-dir experiments\comparison_visuals
