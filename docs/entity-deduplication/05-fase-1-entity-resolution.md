# Entity Resolution Fase 1 — Apa yang Sudah Ditangani

Ringkasan praktis isi [entity_resolution.py](../../hybrid_rag/ingestion/knowledge_graph/entity_resolution.py). Latar belakang, perbandingan package, dan angka benchmark lengkap ada di [00-ringkasan-dan-rekomendasi.md](00-ringkasan-dan-rekomendasi.md).

## 1. Fungsinya

Sebelum entity hasil ekstraksi ditulis ke graph, resolver menyatukan **varian nama dari entity yang sama** menjadi satu nama canonical. Semua aturannya deterministik, tanpa LLM, dan mengutamakan precision: lebih baik ada duplikat daripada dua entity berbeda tergabung.

```
ekstraksi per dokumen
   └─► EntityResolver.resolve()        ← merge_nodes_and_edges, sebelum tulis
         1. alias YAML (bila dokumen memilih entity_registry)
         2. aturan: kunci sama / pola "Long (ABBR)"
         3. catat keputusan ke Postgres, baca ulang, rewrite node + edge
   └─► merge ke Neo4j / VDB seperti biasa (nama sudah canonical)

hapus dokumen → rebuild
   └─► EntityResolver.canonical_names() ← lookup registry saja
```

## 2. Kasus yang ditangani

| Kasus | Contoh | Hasil |
|---|---|---|
| Huruf besar/kecil | `KOTA MALANG` / `Kota Malang`, `MADE ARTANA` / `Made Artana` | Digabung |
| Tanda baca & spasi | `M Jufri Chaniago` / `M. Jufri Chaniago`, `Indeks Kota/Kabupaten Kreatif` / `Indeks Kota / Kabupaten Kreatif` | Digabung |
| Urutan kata (bukan nama lembaga) | `Kota Malang` / `Malang Kota` | Digabung |
| Kata sambung & format nomor | `Badan Riset Dan Inovasi Nasional` / `Badan Riset Inovasi Nasional`, `Perpres No 12 Tahun 2025` / `Perpres Nomor 12 Tahun 2025`, `Strategi Komunikasi & Narasi` / `… Dan Narasi` | Digabung |
| Pola `Bentuk Panjang (SINGKATAN)` | `World Conference On Creative Economy (WCCE)` + `WCCE` → `World Conference On Creative Economy` | Digabung, bila singkatan = huruf awal |
| Alias kustom (YAML) | `ICCN`, `Indonesia Creative Cities Networks` → `Indonesia Creative Cities Network` | Digabung |
| Lintas dokumen | `Malang Kota` di dokumen ke-2 → `Kota Malang` dari dokumen ke-1 | Digabung lewat registry |
| Rebuild setelah delete | Nama mentah di cache ekstraksi → nama canonical di graph | Dipetakan ulang, alias tidak muncul lagi |

## 3. Pengaman (yang sengaja **tidak** digabung)

| Pengaman | Contoh yang tetap terpisah |
|---|---|
| Tipe entity harus sama | `Kota Malang` (location) ≠ `Malang Kota` (organization). Nama yang **persis sama** tetap satu node apa pun tipenya, karena ID node = nama (lihat §6). |
| Angka dibaca dengan pemisahnya, berurutan | `Pasal 9` ≠ `Pasal 10`, `Lampiran 1.2` ≠ `Lampiran 12`, `Pasal 9 Ayat 1` ≠ `Pasal 1 Ayat 9` |
| Nama lembaga: awalan dan urutan kata adalah identitas | `Dinas Pariwisata, Ekonomi Kreatif, Kepemudaan Dan Olahraga` ≠ `Dinas Kepemudaan, Olahraga, Pariwisata, Dan Ekonomi Kreatif`; `Dinas X` ≠ `Suku Dinas X` ≠ `Badan X` |
| Kata pembeda tetap dihitung | `Kota Malang` ≠ `Kabupaten Malang`, `Kementerian Keuangan` ≠ `Kementerian Perdagangan` |
| Singkatan harus huruf awal | `Kota Malang (JATIM)`: "JATIM" bukan alias Kota Malang |
| Singkatan ambigu tidak ditaut | `BPS` tidak ditaut bila dokumen memuat `Badan Pusat Statistik (BPS)` **dan** `Balai Pelestarian Sejarah (BPS)` |
| `never_merge` di YAML | `Kemenparekraf` ≠ `Bekraf`, `Kota Malang` ≠ `Kabupaten Malang` |
| Nama kosong setelah dinormalisasi | `—`, `()` tidak pernah dipakai sebagai kunci |
| Aksara non-Latin dipertahankan | `東京` ≠ `大阪` |
| Entity multimodal tidak disentuh | `… (table)`, `… (image)`, sidecar `drawing/table/equation` |
| Tanpa rantai | `X (A)` dan `A` bergabung ke `X`; tidak ada A~B~C berantai |

## 4. Komponen di kode

| Komponen | Isi |
|---|---|
| `entity_key(name)` | Kunci lookup: NFKD, buang diakritik, casefold, tanda baca → spasi. Tidak pernah ditampilkan. |
| `match_key(name)` | Kunci pencocokan dari `entity_key`: buang kata sambung (`dan`, `no`, `nomor`, `tahun`, `the`, `of`). Nama lembaga → kunci berurutan (`=dinas pariwisata …`). Selain itu kata diurutkan, token berangka tetap berurutan. |
| `split_long_abbr(name)` | Mengenali pola `Long (ABBR)` yang valid (ABBR = inisial). |
| `load_entity_registry(name)` | Memuat `entity_registry/<name>.yaml` (di-cache). Nama yang tidak ada → error beserta daftar yang tersedia. |
| `AliasRegistry` | Tabel Postgres `hybrid_rag_entity_alias` per workspace. Baris `rule` tidak pernah ditimpa (keputusan pertama menang); baris `user` (dari YAML) menimpa. |
| `EntityResolver.resolve()` | Jalur ingest: alias YAML → aturan → simpan keputusan → baca ulang → rewrite node & edge (self-loop baru dibuang). |
| `EntityResolver.canonical_names()` | Jalur rebuild: lookup registry saja, tanpa YAML dan tanpa menulis. |
| `rewrite()` | Memindahkan record node/edge ke nama canonical. |

Canonical dipilih deterministik: nama yang sudah tercatat di registry, lalu bentuk terpanjang. Pada panjang sama, bentuk yang bukan ALL-CAPS menang. Untuk pola `X (A)`, yang dipakai adalah `X`.

## 5. Cara pakai

**Sekali saja.** Jalankan `make db-init` untuk membuat tabel `hybrid_rag_entity_alias`.

**Semua pengaturan lewat parameter ingest, per dokumen:**

```python
await rag.insert_content_list(
    content_list,
    file_path=path,
    entity_resolution=True,    # default True; False = exact-name merge seperti sebelumnya
    entity_registry="iccn",    # opsional: file alias di entity_registry/iccn.yaml
)
```

- Kedua nilai disimpan di metadata doc_status (`entity_resolution`, `kg_entity_registry`), sehingga retry/reprocess memakai pengaturan yang sama.
- Dengan `entity_resolution=False`, `entity_registry` diabaikan, dan saat rebuild nama dari dokumen itu tidak dipetakan ulang.
- Dokumen lama yang belum punya metadata `entity_resolution` dianggap aktif.

**Alias kustom per domain** (opsional), dipilih per dokumen seperti ontology:

```yaml
# hybrid_rag/ingestion/knowledge_graph/entity_registry/iccn.yaml
global:                       # berlaku untuk semua tipe
  - canonical: Indonesia Creative Cities Network
    aliases: [ICCN, Indonesia Creative Cities Networks]
by_type:                      # hanya untuk satu tipe ekstraksi
  program:
    - canonical: Kota Kreatif Indonesia
      aliases: [KKI]
never_merge:                  # tidak pernah digabung, apa pun aturannya
  - [Kemenparekraf, Bekraf]
```

Catatan YAML:
- Pencocokan mengabaikan kapital, tanda baca, kata sambung, dan urutan kata (kecuali nama lembaga), jadi cukup tulis varian yang **kata-katanya** berbeda.
- Alias yang sudah terpakai tercatat per workspace, sehingga dokumen berikutnya tetap mengikutinya walau di-ingest tanpa `entity_registry`.

## 6. Yang belum ditangani

> Sebagian besar baris di bawah kini ditangani **Fase 2** (adjudikasi LLM): lihat [06-fase-2-entity-adjudication.md](06-fase-2-entity-adjudication.md). Tabel ini menggambarkan Fase 1 saja.

| Kasus | Contoh | Ditangani di |
|---|---|---|
| Kualifier/containment | `Dewan Pakar` / `Dewan Pakar ICCN`, `Provinsi Sumatera Utara` / `Sumatera Utara` | Fase 2 (LLM) |
| Salah ketik | `Dadik Wahyu Cang` / `Chang`, `Ahmad Hafidz` / `Hafizh` | Fase 2 |
| Akronim silabik / tanpa pola `(X)` | `Rindekraf`, `Menpora`, `BPS`, `CSR` | YAML sekarang, Fase 2 otomatis |
| Lembaga sama, nama ditulis beda | `Dinas Pariwisata Dan Kebudayaan Kota Malang` / `Disparbud Kota Malang` | Fase 2 |
| Tipe dari LLM tidak konsisten | `ASTA EKONOMI KREATIF` (concept) / `Asta Ekonomi Kreatif` (program) | Fase 2 |
| Homonim nama persis | `Dinas Pariwisata` milik dua kota | Prompt ekstraksi (nama + daerah) |
| Nama yang hanya muncul sebagai ujung relasi | — | Belum direncanakan |
| Graph lama sebelum fitur aktif | — | Fase 3 (backfill) |

## 7. Pengujian

- Unit test: [tests/test_entity_resolution.py](../../tests/test_entity_resolution.py), dijalankan dengan `make test`.
- Benchmark korpus `data/`: [evaluation/entity_dedup_benchmark.py](../../evaluation/entity_dedup_benchmark.py) (`ingest`, `candidates`, `evaluate`) dengan golden set [entity_dedup_golden.csv](../../evaluation/datasets/entity_dedup_golden.csv).
- Hasil terakhir: precision 1,000, recall 0,384, 0 panggilan LLM. Rinciannya di [00 §6.7](00-ringkasan-dan-rekomendasi.md).
