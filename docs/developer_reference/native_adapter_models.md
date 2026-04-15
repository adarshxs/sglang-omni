# Native Adapter Models

This note documents the native-adapter path for models whose request loop still needs model-local side computation even though SGLang owns scheduling, prefill, and decode iteration.

## When to use this path

Use the native-adapter pattern when all of the following are true:

- the checkpoint is not a normal Hugging Face `AutoModel` target
- you want SGLang to drive request scheduling / prefill / decode iteration
- the model still needs request-local side computation outside the standard token-only forward path

VoxCPM2 is the first model using this pattern.

## Contract

The shared runtime stays generic. The model owns the odd behavior.

### Request state

`SGLangARRequestData` can optionally carry a grouped adapter state object:

- `native_adapter.prompt_text`
- `native_adapter.metadata`
- `native_adapter.state`
- `native_adapter.latest_multimodal_outputs`
- `native_adapter.finish_reason`
- `native_adapter.usage`

The grouped type is `NativeAdapterRequestState` in `sglang_omni/models/native_adapter/scaffold.py`.

### Model-side hooks

A native adapter model should implement the `NativeAdapterModel` protocol from `sglang_omni/models/native_adapter/scaffold.py`:

- `set_native_adapter_requests(requests)`
- `clear_native_adapter_requests()`
- `pop_native_adapter_result(request_id)`

The shared `NativeAdapterScaffoldMixin` already provides those methods and request/result bookkeeping.

### Step result

Each decode step can return a `NativeAdapterStepResult`, which carries:

- `token_id`
- `finished`
- `finish_reason`
- `next_input_embeds`
- `multimodal_outputs`
- `usage`
- `extra`

The generic SGLang AR runtime consumes this object and updates request state automatically.

## Runtime behavior

The native-adapter path is implemented directly inside `sglang_omni/engines/omni/runtime/sglang_ar.py`.

The generic runtime now:

- injects active requests into models that satisfy `NativeAdapterModel`
- lets the model return per-step native side results
- stores decode-time feedback embeddings for the next step
- streams multimodal outputs when present
- respects model-provided finish reasons and stop-token controls

This keeps the execution path close to the normal `create_sglang_ar_engine()` flow rather than introducing a second engine stack.

## Raw checkpoint bridge

If the checkpoint already has a `config.json` but it is not a Hugging Face `PretrainedConfig`, add a bridge config.

For VoxCPM2, this is:

- `sglang_omni/models/voxcpm2/hf_config.py`

The bridge config:

- reads the raw checkpoint config
- maps the fields SGLang needs for `ModelConfig`
- writes a synthetic HF-style config file
- points back to the original native model path

This is required because “has a `config.json`” is not the same as “`AutoConfig` + SGLang `ModelConfig` can load it”.

## VoxCPM2 Shape

VoxCPM2 now follows this layout:

- generation model: `sglang_omni/models/voxcpm2/sglang_model.py`
- request builder: `sglang_omni/models/voxcpm2/pipeline/engine_io.py`
- generation stage: `sglang_omni/models/voxcpm2/pipeline/stages.py`
- audio decode stage: `sglang_omni/models/voxcpm2/audio_vae.py`

The generation model:

- loads a synthetic HF config that mirrors the raw VoxCPM2 checkpoint
- runs `base_lm` and `residual_lm` on SGLang's paged attention backend
- owns the in-tree FSQ / LocEnc / LocDiT / AudioVAE components
- performs native prefill / decode side computation
- returns `NativeAdapterStepResult`

## Self-Contained vs Engine-Native

VoxCPM2 is now self-contained in this repo, but it still uses the native-adapter runtime contract because the request loop is not a plain token sampler.

What you get now:

- one shared SGLang AR engine path
- fully in-tree VoxCPM2 side modules
- SGLang-backed MiniCPM4 prefill/decode

What you do not get automatically:

- true paged-KV acceleration for every side module
- optimal batching for every request-local diffusion-heavy step
- streaming waveform decode for free

Those remain phase-2 optimizations after correctness and self-contained execution are in place.
