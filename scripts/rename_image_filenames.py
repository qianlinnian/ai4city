"""批量删除图片文件名中的中文字符。

默认只预览，不会修改文件。确认预览结果后，增加 --apply 才会真正重命名。
"""

import argparse
import re
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


def is_cjk(character):
    """判断字符是否属于常见 CJK 汉字区。"""
    code_point = ord(character)
    return any(
        start <= code_point <= end
        for start, end in (
            (0x3400, 0x4DBF),   # CJK 扩展 A
            (0x4E00, 0x9FFF),   # CJK 基本区
            (0xF900, 0xFAFF),   # CJK 兼容汉字
            (0x20000, 0x2FA1F), # CJK 扩展 B-F、兼容扩展
            (0x30000, 0x3134F), # CJK 扩展 G-H
        )
    )


def cleaned_stem(stem):
    """删除汉字，并清理删除后残留的重复或首尾分隔符。"""
    result = "".join(character for character in stem if not is_cjk(character))
    result = re.sub(r"_+", "_", result)
    return result.strip(" ._-")


def destination_for(image_path):
    new_stem = cleaned_stem(image_path.stem)
    if not new_stem:
        raise ValueError("删除中文后文件名为空")
    return image_path.with_name(new_stem + image_path.suffix)


def windows_path_key(path):
    """按 Windows 不区分大小写的规则比较目标路径。"""
    return str(path.resolve()).casefold()


def collect_operations(root):
    operations = []
    errors = []

    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not any(is_cjk(character) for character in image_path.stem):
            continue

        try:
            destination = destination_for(image_path)
        except ValueError as exc:
            errors.append("{}：{}".format(image_path, exc))
            continue

        if destination != image_path:
            operations.append((image_path, destination))

    # 在执行任何改名之前一次性检查冲突，避免只改了一半。
    destination_sources = {}
    for source, destination in operations:
        key = windows_path_key(destination)
        previous_source = destination_sources.get(key)
        if previous_source is not None and previous_source != source:
            errors.append(
                "多个文件会得到相同名称：{}、{} -> {}".format(
                    previous_source, source, destination
                )
            )
        else:
            destination_sources[key] = source

        if destination.exists() and windows_path_key(destination) != windows_path_key(source):
            errors.append("目标文件已存在：{} -> {}".format(source, destination))

    return operations, errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="递归删除图片文件名中的中文字符；默认仅预览。"
    )
    parser.add_argument("root", type=Path, help="需要处理的图片根目录")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行重命名；不加此参数时仅预览",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        print("错误：目录不存在：{}".format(root), file=sys.stderr)
        return 2

    operations, errors = collect_operations(root)

    if errors:
        print("发现冲突或无效文件名，未修改任何文件：", file=sys.stderr)
        for error in errors:
            print("  - {}".format(error), file=sys.stderr)
        return 2

    if not operations:
        print("没有发现文件名含中文的图片。")
        return 0

    mode = "执行" if args.apply else "预览"
    print("{}：共发现 {} 张需要重命名的图片".format(mode, len(operations)))
    for source, destination in operations:
        print("{} -> {}".format(source, destination.name))

    if not args.apply:
        print("\n以上仅为预览。确认无误后增加 --apply 参数执行。")
        return 0

    for source, destination in operations:
        source.rename(destination)

    print("\n完成：已重命名 {} 张图片。".format(len(operations)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
