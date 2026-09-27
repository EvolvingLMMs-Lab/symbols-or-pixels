"""Forward-FLOPs model for one answer: a generated video (DiT) or a language-model call.

Video (one DiT pass over the latent video; 50 denoising steps x 2 passes for classifier-free guidance):
    tokens * layers * [2 (6 d^2 + 2 d f) + 4 d tokens + 4 d 512]
covering the linear layers and attention over the video tokens and 512 text tokens. The umT5 text encoder and the
VAE are not counted, so the video side is an underestimate.

Language model (one call with T = prompt + completion tokens, image tokens included):
    ViT + 2 N T + 4 d L T (T + 1) / 2,   ViT = 2 N_vit P + 4 d_vit depth P^2
where N is the language-model parameter count, d the hidden size, L the number of full-attention layers (Gated
DeltaNet layers add no quadratic term) and P the number of ViT patches (512 x 512 frame, patch 16: P = 1,024).
N and N_vit are the rounded counts used in the note (for example 27.35B + 0.43B for Qwen3.6-27B).

Usage (as a module):
    from flops import TEXT_MODELS, VIDEO_MODELS, text_call_flops, video_flops
"""

VIDEO_STEPS, VIDEO_PASSES_PER_STEP, VIDEO_TEXT_TOKENS = 50, 2, 512

# Wan2.2 DiT shapes at 512 x 512 x 81 frames. TI2V-5B: VAE 16x16x4 + patch 1x2x2 -> 16 x 16 x 21 = 5,376 tokens.
# I2V-A14B: VAE 8x8x4 + patch 1x2x2 -> 32 x 32 x 21 = 21,504 tokens; one 14B expert is active per step.
VIDEO_MODELS = {
    "g5": dict(checkpoint="Video-Reason/VBVR-Pro-Wan2.2-TI2V-5B", tokens=5376, layers=30, d=3072, f=14336),
    "g27": dict(checkpoint="Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B", tokens=21504, layers=40, d=5120, f=13824),
}

# Hidden size and full-attention layers from each model's config.json (text_config); ViT width/depth from
# vision_config; patch 16 at 512 x 512.
TEXT_MODELS = {
    "qwen3.5-4b": dict(model="Qwen/Qwen3.5-4B", n=4.36e9, d=2560, full_layers=8, n_vit=0.30e9, d_vit=1024, depth=24),
    "qwen3.5-9b": dict(model="Qwen/Qwen3.5-9B", n=9.22e9, d=4096, full_layers=8, n_vit=0.43e9, d_vit=1152, depth=27),
    "qwen3.6-27b": dict(model="Qwen/Qwen3.6-27B", n=27.35e9, d=5120, full_layers=16, n_vit=0.43e9, d_vit=1152,
                        depth=27),
}
VIT_PATCHES = (512 // 16) ** 2


def dit_pass_flops(tokens, layers, d, f, text_tokens=VIDEO_TEXT_TOKENS):
    return tokens * layers * (2 * (6 * d * d + 2 * d * f) + 4 * d * tokens + 4 * d * text_tokens)


def video_flops(name):
    m = VIDEO_MODELS[name]
    return VIDEO_STEPS * VIDEO_PASSES_PER_STEP * dit_pass_flops(m["tokens"], m["layers"], m["d"], m["f"])


def vit_flops(name, patches=VIT_PATCHES):
    m = TEXT_MODELS[name]
    return 2 * m["n_vit"] * patches + 4 * m["d_vit"] * m["depth"] * patches ** 2


def text_call_flops(name, prompt_tokens, completion_tokens):
    m = TEXT_MODELS[name]
    t = prompt_tokens + completion_tokens
    return vit_flops(name) + 2 * m["n"] * t + 4 * m["d"] * m["full_layers"] * t * (t + 1) / 2
