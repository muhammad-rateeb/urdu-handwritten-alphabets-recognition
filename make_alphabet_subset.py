from pathlib import Path

from PIL import Image


def main():
    root = Path(r"c:/Users/Administrator/Desktop/BSCS/6th semester/ml/ml_ccp/dataset")
    out = Path(r"c:/Users/Administrator/Desktop/BSCS/6th semester/ml/ml_ccp/alphabets_128x128")
    out.mkdir(exist_ok=True)

    classes = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: int(p.name))
    for class_dir in classes:
        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if not files:
            continue
        src = files[0]
        img = Image.open(src).convert("L").resize((128, 128))
        img.save(out / f"{class_dir.name}.png")

    print("saved", len(list(out.glob("*.png"))))


if __name__ == "__main__":
    main()
