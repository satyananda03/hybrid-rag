### TO DO :
1. Benchmarking amazon titan text embedding v2 vs amazon nova multimodal embedding v1
2. Benchmarking NDCG & MRR metrics on amazon bedrock reranker 
3. Metadata filtering for graph search

### Parameter Ingestion

`rag.insert_content_list(content_list, file_path=..., ...)`

| Parameter | Tipe | Default | Fungsi |
| --- | --- | --- | --- |
| `content_list` | `list[dict]` | wajib | Blok hasil `parse_file` (teks, gambar, tabel, persamaan) |
| `file_path` | `str` / `Path` | `"unknown_document"` | Nama file untuk sitasi |
| `doc_id` | `str` | dibuat dari isi dokumen | ID dokumen; isi yang sama menghasilkan ID yang sama |
| `ontology` | `str` | `None` | File di `knowledge_graph/ontology/` (`.ttl` / `.owl`). Ekstraksi dipandu dan divalidasi skemanya. `None` memakai kontrak bawaan tanpa validasi. File yang tidak ada langsung error |
| `entity_registry` | `str` | `None` | File di `knowledge_graph/entity_registry/` (`.yaml`) berisi alias dan pasangan `never_merge`. `None` hanya memakai aturan deterministik. File yang tidak ada langsung error |
| `entity_resolution` | `bool` | `True` | Gabungkan varian nama entity dokumen ini sebelum ditulis ke graph. `False` memakai merge nama persis, dan `entity_registry` diabaikan |
| `entity_resolution_llm` | `bool` | `True` | LLM ikut menilai pasangan nama yang tidak bisa diputuskan aturan. Hanya berlaku bila `entity_resolution=True`; kalau panggilan LLM gagal, hasil aturan tetap dipakai |
| `orphan_repair` | `bool` | `True` | Untuk chunk yang punya entity tanpa relasi, satu panggilan LLM tambahan mencari relasi yang memang tertulis di teks. Biaya hanya di chunk tersebut |
| `prune_isolated` | `bool` | `True` | Di akhir batch, hapus entity dokumen ini yang tetap tanpa relasi (node image/table/equation dikecualikan). Teksnya tetap bisa dicari lewat chunk |
| `save_parent` | `bool` | `True` | Simpan parent chunk di samping child yang di-embed, supaya retrieval bisa mengembalikan passage utuh (`return_parent`). Matikan hanya untuk dokumen yang tidak akan jadi bukti |

- Pengaturan dari `ontology` sampai `save_parent` disimpan di metadata dokumen, jadi retry atau resume dokumen tetap memakai nilai aslinya.
- Parameter berlaku per dokumen, kecuali `prune_isolated`: penghapusan berjalan sekali di akhir batch, dan hanya mencakup dokumen yang diingest dengan `prune_isolated=True`.
- Ukuran chunk dan konkurensi bukan parameter pemanggilan, melainkan ENV: `PARENT_CHUNK_SIZE` (parent = satu section dokumen; hanya section yang lebih besar dipotong di antara paragraf), `CHILD_CHUNK_SIZE`, `CHILD_CHUNK_OVERLAP`, `MAX_GLEANING`, `MAX_EXTRACT_INPUT_TOKENS`, `MAX_PARALLEL_INGESTION`, `EMBEDDING_BATCH_NUM`.
- `parse_file(file_path)` hanya menerima path file dan mengembalikan `content_list`.

```python
async with parse_file(path) as content_list:
    async with write_pool.lease("workspace_h") as rag:
        await rag.insert_content_list(
            content_list,
            file_path=path,
            doc_id="doc-12",
            ontology=None,
            entity_registry="iccn",
            entity_resolution=True,
            entity_resolution_llm=True,
            orphan_repair=True,
            prune_isolated=True,
            save_parent=True,
        )
```

### Melihat Knowledge Graph

Penampil graph (halaman React + JSON API) dipasang ke aplikasi FastAPI lewat `mount_dashboard`. Contoh lengkapnya ada di [`examples/kg_viewer.py`](examples/kg_viewer.py); jalankan dari root repo supaya `.env` terbaca:

```bash
uv run --extra ingestion --extra dashboard --with uvicorn \
    uvicorn kg_viewer:app --app-dir examples
```

Lalu buka <http://127.0.0.1:8000/kg/> dan pilih workspace di halaman itu. Intinya hanya tiga baris:

```python
from hybrid_rag.dashboard import mount_dashboard

app = FastAPI()
mount_dashboard(app, prefix="/kg")   # penampil di /kg/, API di /kg/api/*
```

- Endpoint: `/kg/api/workspaces`, `/kg/api/graph?workspace=...`, `/kg/api/labels/popular`, `/kg/api/labels/search`.
- `uvicorn` tidak termasuk extra `dashboard` (server dibawa oleh aplikasi pemilik), jadi contoh memakai `--with uvicorn`.
- Pool retrieval membangun fungsi embedding saat dibuat, jadi `providers.configure(get_embeddings=...)` tetap wajib walau penampil tidak memanggilnya.
- API membuka graph semua workspace tanpa autentikasi. Uvicorn mendengarkan `127.0.0.1` secara bawaan; bila dibuka ke jaringan, pasang autentikasi lewat `mount_dashboard(app, dependencies=[...])`.
- Halaman statisnya sudah ter-build di `hybrid_rag/dashboard/static`. Bila perlu build ulang: `bun run build` di folder `dashboard/`.

Terminal Command :
```uv run --extra ingestion --extra dashboard --with uvicorn uvicorn kg_viewer:app --app-dir examples
```

### Membuat Entity Registry Baru

Buat file `hybrid_rag/ingestion/knowledge_graph/entity_registry/<nama>.yaml`, lalu pakai dengan `entity_registry="<nama>"` saat ingest. Nama yang tidak ada akan error dan menampilkan daftar registry yang tersedia. Contoh lengkap: `entity_registry/iccn.yaml`.

```yaml
global:                       # alias untuk semua tipe entity
  - canonical: Indonesia Creative Cities Network
    aliases: [ICCN, Jejaring Kota Kreatif Indonesia]

by_type:                      # alias khusus satu tipe entity (opsional)
  program:
    - canonical: Kota Kreatif Indonesia
      aliases: [KKI]

never_merge:                  # pasangan yang tidak boleh digabung (opsional)
  - [Kota Malang, Kabupaten Malang]
```

- Huruf kapital, tanda baca, dan urutan kata diabaikan saat mencocokkan, jadi tulis varian hanya bila katanya berbeda.
- Satu alias tidak boleh menunjuk ke dua `canonical` berbeda (error saat load).