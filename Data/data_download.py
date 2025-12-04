from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wayrise/DexGarmentLab",
    allow_patterns=["Fling_Dress.zip", "Fling_Tops.zip", "Fling_Trousers.zip", "Fold_Dress.zip", "Fold_Tops.zip", "Fold_Trousers.zip", 
                    "Hang_Dress.zip", "Hang_Tops.zip", "Hang_Trousers.zip", "Hang_Coat.zip", "Store_Tops.zip", "Wear_Baseballcap.zip", 
                    "Wear_Bowlhat.zip", "Wear_Scarf.zip"],
    local_dir="./Data",
    repo_type="dataset",
    resume_download=True,
)