"""Validate the complete Fig2.tif through Fig9.tif PLOS export set."""

from __future__ import annotations

from paper_style import PLOS_TIFF_DIR, validate_plos_tiff


def main() -> None:
    results = [validate_plos_tiff(PLOS_TIFF_DIR / f"Fig{number}.tif") for number in range(2, 10)]
    columns = [
        "filename", "width_px", "height_px", "dpi", "color_mode",
        "has_alpha", "compression", "file_size_MB", "status",
    ]
    widths = {column: max(len(column), *(len(str(row[column])) for row in results)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in results:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    main()
