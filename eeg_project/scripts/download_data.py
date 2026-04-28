import os
import argparse
import requests


def download_file(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        chunk = 1024 * 1024
        downloaded = 0
        with open(out_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    downloaded += len(part)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r已下载: {downloaded/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="")
    print("\n下载完成:", out_path)


def main():
    parser = argparse.ArgumentParser(description="下载 EEG 原始 .fif 文件到项目结构")
    parser.add_argument("--url", required=True, help="源文件直链 URL（HTTP/HTTPS）")
    parser.add_argument(
        "--subject",
        default="01",
        help="受试者编号，将保存为 eeg_project/data/raw/sub{subject}/eeg/sub{subject}_taskimagine_eeg.fif",
    )
    args = parser.parse_args()

    out_path = os.path.join(
        "eeg_project",
        "data",
        "raw",
        f"sub{args.subject}",
        "eeg",
        f"sub{args.subject}_taskimagine_eeg.fif",
    )
    download_file(args.url, out_path)


if __name__ == "__main__":
    main()


