# VoxCPM2 TTS Usage

This guide covers the first VoxCPM2 integration milestone in `sglang-omni`.

Current scope:

- Plain text -> speech
- 48 kHz waveform output
- Native VoxCPM2 side computation behind an SGLang-backed scaffold loop

Not included yet:

- Reference-audio voice cloning
- Prompt-audio continuation
- Incremental streaming decode
- Deep paged-KV / batched native-side optimization

## Prerequisites

Install the upstream `voxcpm` package, or point `sglang-omni` at a source checkout:

```bash
# Option A
pip install voxcpm

# Option B
export SGLANG_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM
```

## Launch

```bash
sgl-omni serve \
  --model-path openbmb/VoxCPM2 \
  --config examples/configs/voxcpm2_tts.yaml \
  --port 8000
```

## Basic Request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a VoxCPM2 test."
  }' \
  --output output.wav
```

## Optional Generation Controls

The first implementation exposes VoxCPM2-native generation knobs through stage params on the `generation` stage:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a VoxCPM2 test.",
    "max_new_tokens": 512,
    "stage_params": {
      "generation": {
        "min_len": 2,
        "cfg_value": 2.0,
        "inference_timesteps": 10
      }
    }
  }' \
  --output output.wav
```

## Notes

- The pipeline currently loads only a lightweight AudioVAE stage separately; the full native VoxCPM2 stack lives in the generation stage.
- The native side still runs request-local residual LM + diffusion logic, so this first milestone is aimed at correctness and future extensibility rather than peak throughput.

## Follow-Up Optimization Path

The compatibility-first integration is intentionally a stepping stone. The natural next optimizations are:

- Replace the scaffold-only MiniCPM path with a truly optimized paged-KV MiniCPM4 execution path.
- Remove the current batch-1 native-side assumption by isolating request-local residual LM / diffusion state more cleanly.
- Add sliding-window or incremental AudioVAE decode so streaming does not require full waveform re-decode.
- Revisit CUDA graph and overlap settings once the native-side execution path is stable.
