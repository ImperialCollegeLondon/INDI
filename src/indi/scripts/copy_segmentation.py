import argparse
import pathlib
import shutil


def get_arguments():
    parser = argparse.ArgumentParser(description="Copy segmentation images")

    parser.add_argument(
        "--path",
        type=pathlib.Path,
        required=True,
        help="Path to data folders",
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        required=True,
        help="Path to save the output",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    # Get all folders containing diffusion images

    folders = args.path.glob("*/**/Python_post_processing/")

    for folder in folders:

        folder_name = folder.relative_to(args.path)
        output_session = args.out_path / folder_name.parent / "Python_post_processing" / "session"
        output_session.mkdir(parents=True, exist_ok=True)

        session_folder = folder / "session"

        shutil.copy(
            session_folder / "manual_lv_segmentation_slice_000.npz",
            output_session / "manual_lv_segmentation_slice_000.npz",
        )
        shutil.copy(session_folder / "u_net_segmentation.npz", output_session / "u_net_segmentation.npz")
