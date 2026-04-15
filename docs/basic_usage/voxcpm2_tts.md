# VoxCPM2 TTS Usage

This guide covers the native VoxCPM2 integration in `sglang-omni`.

Current scope:

- Plain text -> speech
- Reference-audio voice cloning
- Prompt-audio continuation
- 48 kHz waveform output
- Native VoxCPM2 side modules in-tree
- SGLang-backed MiniCPM4 base/residual LM execution

Not included yet:

- Incremental streaming decode
- OpenAI speech endpoint wiring specific to VoxCPM2 request extras

## Prerequisites

Install the normal `sglang-omni` dependencies. No separate `voxcpm` package or source checkout is required anymore.

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

VoxCPM2 exposes generation knobs through stage params on the `generation` stage:

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

- The pipeline keeps the three-stage layout: preprocessing -> latent generation -> audio decode.
- Generation runs a native VoxCPM2 model class that owns the SGLang-backed MiniCPM4 base/residual LMs plus the in-tree FSQ / LocEnc / LocDiT stack.
- Audio decode still happens in a separate stage so latent patches remain the interface between generation and waveform reconstruction.

## Follow-Up Optimization Path

The native port is now correctness-first but self-contained. The natural next optimizations are:

- Tune the batched VoxCPM2 request path and scheduler limits for larger concurrent decode loads.
- Add selective `torch.compile` / CUDA graph capture to the diffusion-heavy parts once runtime behavior is stable.
- Add sliding-window or incremental AudioVAE decode so streaming does not require full waveform re-decode.
