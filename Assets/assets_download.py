from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wayrise/DexGarmentLab",
    allow_patterns=["Garment.zip", "Robots.zip", "Scene.zip", "LeapMotion.zip", "Human.zip"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
)