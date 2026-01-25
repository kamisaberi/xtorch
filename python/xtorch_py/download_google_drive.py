import sys
import argparse
import os
import gdown


def main():
    USER_AGENT = "pytorch/vision"
    parser = argparse.ArgumentParser(
        description="Download File from Google Drive using gdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("file_id", type=str, help="Google Drive File ID")
    parser.add_argument("--md5", type=str, default="00000000000000000000000000000000", help="File MD5 Hash")
    parser.add_argument("--output", type=str, default="./", help="Path to store downloaded file")

    args = parser.parse_args()
    gdown.download(id=args.file_id, output=args.output, quiet=False, user_agent=USER_AGENT)


if __name__ == "__main__":
    main()
