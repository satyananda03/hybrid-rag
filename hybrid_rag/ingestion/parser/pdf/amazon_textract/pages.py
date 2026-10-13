"""PDF → halaman: split per halaman untuk Textract, render hanya bila butuh piksel."""
import pymupdf as fitz
from PIL import Image, ImageFilter

RENDER_DPI = 400


def enhance_image(
    pil_img: Image.Image,
    apply_otsu: bool = False,
    denoise: bool = True,
) -> Image.Image:
    """
    Tingkatkan kualitas gambar menggunakan PIL (Pillow).
    Mengubah gambar ke grayscale dan menerapkan Unsharp Masking.
    """
    # 1. Konversi ke Grayscale
    gray = pil_img.convert("L")

    # 2. Denoising & Sharpening (Unsharp Masking bawaan PIL)
    if denoise:
        gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    # 3. Otsu Binarization (Opsional, bawaan PIL via point/threshold)
    if apply_otsu:
        gray = gray.point(lambda p: 255 if p > 128 else 0)

    # Tetap grayscale: Textract menerima PNG mode "L"; konversi ke RGB
    # cuma menggandakan ukuran PNG & waktu encode tanpa mengubah hasil OCR.
    return gray


def render_page_image(pdf_path: str, page_idx: int, dpi: int = RENDER_DPI) -> Image.Image:
    """Render satu halaman → PIL grayscale ter-enhance (piksel sama seperti render lama).

    Hanya dipakai untuk crop figure dan fallback halaman >10 MB; teks/tabel
    dibaca Textract langsung dari PDF per halaman.
    """
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        return enhance_image(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    finally:
        doc.close()


def split_pdf_pages(pdf_path: str, dpi: int = RENDER_DPI):
    """Pecah PDF jadi PDF satu halaman tanpa rasterisasi (~ms/halaman).

    Sesuai best practice Textract ("don't convert or downsample"). ``img_w``/``img_h``
    tetap ukuran piksel render ``dpi`` (dihitung dari geometri, tanpa render) supaya
    filter ukuran figure & crop di ``reconstruct_content_list`` tidak berubah.
    """
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        for page_idx, page in enumerate(doc):
            one = fitz.open()
            one.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            # /Rotate dibakar ke konten (tampilan sama): Textract tidak perlu menafsirkan
            # rotasi, orientasi bbox-nya = orientasi render yang dipakai crop figure.
            one[0].remove_rotation()
            pdf_bytes = one.tobytes(garbage=3, deflate=True)
            one.close()
            irect = (page.rect * mat).irect  # == ukuran get_pixmap(matrix=mat)
            yield {
                "page_idx": page_idx,
                "source_pdf": pdf_path,
                "doc_bytes": pdf_bytes,
                "img_w": irect.width,
                "img_h": irect.height,
            }
    finally:
        doc.close()
