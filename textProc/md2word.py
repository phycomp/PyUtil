#!/usr/bin/env python
import os
from pathlib import Path
from sys import argv
from pypandoc import convert_file
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


def 標楷體(docx_path):
    """確保 Word 文件中的所有樣式預設中文字型皆為標楷體"""
    doc = Document(docx_path)

    # 修改預設樣式 (Normal)
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"  # 西文字型
    font.size = Pt(12)  # 12pt (小四)

    # 設定中文字型 (East Asia font) 為標楷體
    rPr = style._element.get_or_add_rPr()
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "標楷體")  # 指定標楷體
    rPr.append(rFonts)

    doc.save(docx_path)


def 轉換格式(md_filename, output_docx):
    print(f"正在轉換 {md_filename} 至 {output_docx}...")

    # 定義 Pandoc 轉換參數
    extra_args = [
        # "--highlight-style=tango",  # 程式碼高亮樣式
        "--syntax-highlighting=tango",
        # "--toc",  # 自動生成學術目錄（可選）
        "--number-sections",  # 自動為章節標題編號（1. 1.1 等學術規範）
    ]

    # 執行轉換
    convert_file(
        md_filename, "docx", outputfile=output_docx, extra_args=extra_args
    )

    # 後處理：強制套用標楷體字型設定
    標楷體(output_docx)
    print("轉換完成！已成功設定標楷體與學術樣式。")


if __name__ == "__main__":
    # 測試用 MD 檔案名稱
    # md_file = "sample.md"
    輸入=Path(argv[1])
    # print(dir(輸入))
    # ['__bytes__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__firstlineno__', '__format__', '__fspath__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rtruediv__', '__setattr__', '__sizeof__', '__slots__', '__static_attributes__', '__str__', '__subclasshook__', '__truediv__', '_copy_from', '_copy_from_file', '_copy_from_symlink', '_delete', '_drv', '_filter_trailing_slash', '_format_parsed_parts', '_from_dir_entry', '_from_parsed_parts', '_from_parsed_string', '_hash', '_info', '_parse_path', '_parse_pattern', '_parts_normcase', '_parts_normcase_cached', '_raw_path', '_raw_paths', '_remove_leading_dot', '_remove_trailing_slash', '_root', '_str', '_str_normcase', '_str_normcase_cached', '_tail', '_tail_cached', 'absolute', 'anchor', 'as_posix', 'as_uri', 'chmod', 'copy', 'copy_into', 'cwd', 'drive', 'exists', 'expanduser', 'from_uri', 'full_match', 'glob', 'group', 'hardlink_to', 'home', 'info', 'is_absolute', 'is_block_device', 'is_char_device', 'is_dir', 'is_fifo', 'is_file', 'is_junction', 'is_mount', 'is_relative_to', 'is_reserved', 'is_socket', 'is_symlink', 'iterdir', 'joinpath', 'lchmod', 'lstat', 'match', 'mkdir', 'move', 'move_into', 'name', 'open', 'owner', 'parent', 'parents', 'parser', 'parts', 'read_bytes', 'read_text', 'readlink', 'relative_to', 'rename', 'replace', 'resolve', 'rglob', 'rmdir', 'root', 'samefile', 'stat', 'stem', 'suffix', 'suffixes', 'symlink_to', 'touch', 'unlink', 'walk', 'with_name', 'with_segments', 'with_stem', 'with_suffix', 'write_bytes', 'write_text']
    輸出=輸入.with_suffix('.docx')
    # output_file = "academic_report.docx"

    if os.path.exists(輸入):
        轉換格式(輸入, 輸出)
    else:
        print(f"找不到檔案：{md_file}，請先建立測試用的 Markdown 檔案。")
