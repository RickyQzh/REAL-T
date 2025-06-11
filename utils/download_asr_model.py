import argparse
import os
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ASR model from Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="FireRedTeam/FireRedASR-AED-L",
                        help="Hugging Face repo ID, e.g., FireRedTeam/FireRedASR-AED-L")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the downloaded model (relative or absolute path)")

    args = parser.parse_args()

    save_path = os.path.abspath(args.save_dir)
    save_path = os.path.join(save_path, args.repo_id.split("/")[-1])
    os.makedirs(save_path, exist_ok=True)

    snapshot_download(repo_id=args.repo_id, local_dir=save_path)

    print(f"Model from '{args.repo_id}' downloaded to '{save_path}'")
