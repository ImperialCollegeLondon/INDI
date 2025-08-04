import argparse
import pathlib


def get_arguments():
    parser = argparse.ArgumentParser(description="Delete empty folders")

    parser.add_argument(
        "path",
        type=pathlib.Path,
        help="Path to the folder to clean up",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    path = args.path

    if not path.exists():
        print(f"The specified path {path} does not exist.")
        return

    # folders = list(args.path.glob("*/**/diffusion_images/"))
    folders = list(args.path.glob("*/**/Python_post_processing/"))
    # folders = list(args.path.rglob("*"))
    print(f"Found {len(folders)} folders to check for emptiness.")

    for folder in folders:
        if not folder.is_dir():
            continue

        # if not (folder.parent / "diffusion_images").exists():
        #     print(f"Deleating empty folder: {folder.parent}")
        #     for root, dirs, files in folder.parent.walk(top_down=False):
        #         for name in files:
        #             (root / name).unlink()
        #         for name in dirs:
        #             (root / name).rmdir()

        # if not any(folder.iterdir()) and not "Python" in str(folder):
        #     print(f"Folder {folder} is empty, deleting it.")
        #     folder.rmdir()
        #     print(f"Deleted empty folder: {folder}")

        denoising_folders = (folder / "..").glob("*_no_denoising")

        denoising_folders = sorted(denoising_folders, key=lambda x: x.name)

        if len(denoising_folders) > 1:
            print(f"Deleting repeated folder: {folder.parent}")
            for denoising_folder in denoising_folders[:-1]:
                print(f"Deleting folder: {denoising_folder}")
                # delete all files and folders inside the denoising folder
                for root, dirs, files in denoising_folder.walk(top_down=False):
                    for name in files:
                        (root / name).unlink()
                    for name in dirs:
                        (root / name).rmdir()

                denoising_folder.rmdir()


if __name__ == "__main__":
    main()
