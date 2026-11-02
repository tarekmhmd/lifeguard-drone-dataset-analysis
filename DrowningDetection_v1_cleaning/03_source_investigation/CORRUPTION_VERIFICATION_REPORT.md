# Corruption Verification Report — contact_sheet_4 sample

Date: 2026-11-02

## Purpose

Check whether the original images corresponding to `contact_sheet_4.png` are actually corrupted or black, or whether any dark appearance was introduced by contact-sheet generation.

## Method (read-only)

- Each original image was read directly from its ZIP archive and decoded with PIL.
- File size, decoded dimensions, and mean RGB brightness were recorded.

## Results

All 12 images decode successfully at 640×640 with non-zero mean brightness. None are black or corrupted.

| Image | Bytes | Size | Mean RGB | Verdict |
|---|---:|---:|---:|---|
| `10_MOV-52_jpg.rf.22538ea0...` | 15,794 | 640×640 | (181.8, 205.6, 224.4) | OK |
| `129_png_jpg.rf.97afceef...` | 39,139 | 640×640 | (122.0, 126.5, 139.4) | OK |
| `203512036-1-_jpg.rf.d8b66eff...` | 60,896 | 640×640 | (75.2, 151.2, 165.2) | OK |
| `0000001_02999_d_0000005_jpg.rf.64f47ee0...` | 106,979 | 640×640 | (120.6, 130.1, 112.8) | OK |
| `0000242_00627_d_0000003_jpg.rf.f613ba88...` | 63,645 | 640×640 | (109.5, 107.4, 99.8) | OK |
| `0000242_02762_d_0000010_jpg.rf.cf37ead6...` | 69,800 | 640×640 | (105.2, 104.6, 104.8) | OK |
| `KakaoTalk_20230504_000756264_04_jpg.rf.ce823e72...` | 37,526 | 640×640 | (132.2, 197.5, 209.8) | OK |
| `KakaoTalk_20230504_000756264_10_jpg.rf.cc16880d...` | 64,290 | 640×640 | (94.3, 107.2, 106.3) | OK |
| `KakaoTalk_20230504_000756264_16_jpg.rf.ed8afc46...` | 49,715 | 640×640 | (112.5, 122.6, 124.5) | OK |
| `10687_jpg.rf.edc349e5...` | 66,685 | 640×640 | (95.3, 130.0, 131.5) | OK |
| `11159_jpg.rf.43800a60...` | 62,277 | 640×640 | (101.4, 134.6, 134.9) | OK |
| `130163_jpg.rf.4ea8236d...` | 17,230 | 640×640 | (50.3, 119.7, 154.7) | OK |

## Conclusion

**No corrupted or black original images** were found in the sheet-4 sample. Any dark/black appearance is not present in the source files. No source file was modified.
