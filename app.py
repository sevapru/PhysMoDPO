"""
PhysMo Studio -- Gradio UI for PhysMoDPO
Port 4565

Accepts text prompt + optional spatial controls, runs DPO-optimized
sample.generate from OmniControl/, displays motion MP4 and SMPL params.
"""
import os
import sys
import glob
import time
import subprocess
import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
OMNICONTROL_DIR = os.path.join(WORKSPACE, "OmniControl")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models/physmodpo")
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "/outputs")

JOINT_MAP = {
    "pelvis (0) — root position": 0,
    "left_knee (5)": 5,
    "right_knee (6)": 6,
    "left_ankle (10)": 10,
    "right_ankle (11)": 11,
    "head (15)": 15,
    "left_wrist (20)": 20,
    "right_wrist (21)": 21,
}

COND_MODES = {
    "text + spatial": "both_text_spatial",
    "spatial only": "only_spatial",
    "text only": "only_text",
}

CHECKPOINTS_HELP = {
    "DPO_hml3d_original.pt": "DPO-tuned model on HumanML3D (original prompt format)",
    "DPO_omomo_original.pt": "DPO-tuned model on OMOMO dataset",
    "pretraining_smpl_original.pt": "Base pretrained SMPL model (no DPO)",
    "DPO_hml3d_cross.pt": "DPO cross-dataset model (HumanML3D)",
    "DPO_omomo_cross.pt": "DPO cross-dataset model (OMOMO)",
}


def list_checkpoints() -> list[str]:
    patterns = [
        f"{MODELS_DIR}/**/*.pt",
        f"{MODELS_DIR}/*.pt",
        f"{OMNICONTROL_DIR}/save/**/*.pt",
    ]
    ckpts = []
    for p in patterns:
        ckpts.extend(glob.glob(p, recursive=True))
    if not ckpts:
        return ["(no checkpoint — download from SharePoint, see HUMAN_ACTIONS.md)"]
    return sorted(ckpts)


def checkpoint_info(path: str) -> str:
    name = os.path.basename(path)
    return CHECKPOINTS_HELP.get(name, "Custom checkpoint")


def generate_motion(
    text_prompt: str,
    model_path: str,
    control_joint: str,
    density: int,
    cond_mode: str,
    num_repetitions: int,
    guidance_param: float,
    motion_length: float,
    seed: int,
) -> tuple:
    if not text_prompt.strip() and cond_mode != "spatial only":
        return None, None, "Error: text prompt required unless spatial-only mode."
    if not model_path or not os.path.exists(model_path):
        return None, None, f"Error: checkpoint not found: {model_path}"

    joint_idx = JOINT_MAP.get(control_joint, 0)
    mode_val = COND_MODES.get(cond_mode, "both_text_spatial")

    run_id = f"run_{int(time.time())}"
    out_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # PhysMoDPO's generate script lives under OmniControl/
    cmd = [
        sys.executable, "-m", "sample.generate",
        "--model_path", model_path,
        "--output_dir", out_dir,
        "--text_prompt", text_prompt if text_prompt.strip() else "predefined",
        "--control_joint", str(joint_idx),
        "--density", str(density),
        "--cond_mode", mode_val,
        "--num_repetitions", str(num_repetitions),
        "--guidance_param", str(guidance_param),
        "--motion_length", str(motion_length),
        "--seed", str(seed),
        "--num_samples", "1",
        "--batch_size", "1",
    ]

    env = os.environ.copy()
    # OmniControl subdir must be on PYTHONPATH for module imports
    pythonpath = f"{OMNICONTROL_DIR}:{WORKSPACE}"
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=OMNICONTROL_DIR, env=env
    )

    log = result.stdout[-3000:] if result.stdout else ""
    if result.returncode != 0:
        log += f"\nSTDERR:\n{result.stderr[-2000:]}"
        return None, None, log

    mp4_files = sorted(glob.glob(f"{out_dir}/**/*.mp4", recursive=True))
    npy_files = sorted(glob.glob(f"{out_dir}/**/results.npy", recursive=True))

    mp4 = mp4_files[0] if mp4_files else None
    npy = npy_files[0] if npy_files else None

    log += f"\nOutputs: {out_dir}"
    return mp4, npy, log


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="PhysMo Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# PhysMo Studio\n"
        "**Physically-plausible humanoid motion with DPO optimization**  \n"
        "Based on [PhysMoDPO](https://arxiv.org/abs/2603.13228) — integrates Whole-Body Controller "
        "into diffusion training, then optimizes with DPO for physics plausibility."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            text_prompt = gr.Textbox(
                label="Motion description",
                placeholder="a person walks forward and picks up an object",
                lines=2,
            )
            model_path = gr.Dropdown(
                label="Checkpoint",
                choices=list_checkpoints(),
                value=list_checkpoints()[0],
                allow_custom_value=True,
            )
            ckpt_info = gr.Textbox(label="Checkpoint info", interactive=False, lines=1)
            control_joint = gr.Dropdown(
                label="Control joint (spatial guidance)",
                choices=list(JOINT_MAP.keys()),
                value="pelvis (0) — root position",
            )
            density = gr.Slider(
                0, 100, value=100, step=5,
                label="Spatial control density (%)",
            )
            cond_mode = gr.Radio(
                label="Conditioning mode",
                choices=list(COND_MODES.keys()),
                value="text + spatial",
            )

            with gr.Accordion("Advanced", open=False):
                num_repetitions = gr.Slider(1, 5, value=1, step=1, label="Repetitions")
                guidance_param = gr.Slider(1.0, 5.0, value=2.5, step=0.5, label="Guidance scale")
                motion_length = gr.Slider(2.0, 9.8, value=6.0, step=0.2, label="Motion length (s)")
                seed = gr.Number(value=10, label="Seed", precision=0)

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            video_out = gr.Video(label="Generated motion")
            npy_out = gr.File(label="Download results.npy")
            status_out = gr.Textbox(label="Status / log", lines=8, max_lines=20)

    model_path.change(
        checkpoint_info,
        inputs=[model_path],
        outputs=[ckpt_info],
    )

    refresh_btn = gr.Button("Refresh checkpoint list", size="sm")
    refresh_btn.click(
        lambda: gr.Dropdown(choices=list_checkpoints()),
        outputs=[model_path],
    )

    generate_btn.click(
        generate_motion,
        inputs=[
            text_prompt, model_path, control_joint, density, cond_mode,
            num_repetitions, guidance_param, motion_length, seed,
        ],
        outputs=[video_out, npy_out, status_out],
    )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--server_name", default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 4565)))
    args = p.parse_args()
    demo.launch(server_name=args.server_name, server_port=args.server_port)
