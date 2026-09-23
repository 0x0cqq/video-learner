"""离线重建一次已完成转换或文字修订，验证固定模型结果后的业务链路。"""

import argparse
from pathlib import Path

from video_learner.workflows.replay import replay_conversion, replay_revision


def main() -> None:
    """只读验证输入、草稿和生成版本，不请求模型或修改原产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--revision", help="离线重建指定的文字修订版本，例如 r002")
    args = parser.parse_args()
    book = (
        replay_revision(args.workdir, args.revision)
        if args.revision
        else replay_conversion(args.workdir)
    )
    print(f"离线重建通过：{len(book.chapters)} 章")


if __name__ == "__main__":
    main()
