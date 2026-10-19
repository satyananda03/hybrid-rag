# Entity Resolution Fase 2 — Kandidat, Veto, dan Adjudikasi LLM

Ringkasan praktis [entity_adjudication.py](../../hybrid_rag/ingestion/knowledge_graph/entity_adjudication.py), lanjutan dari [05-fase-1-entity-resolution.md](05-fase-1-entity-resolution.md). Desain dan bukti: [00 §6.2 L3–L6](00-ringkasan-dan-rekomendasi.md).

## 1. Masalah yang diselesaikan

Fase 1 hanya menggabungkan nama yang kuncinya sama. Recall-nya 0,37 di golden set, karena kasus berikut lolos:

| Pola | Contoh |
|---|---|
| Kualifier/containment | `Dewan Pakar` / `Dewan Pakar ICCN`, `Maluku` / `Provinsi Maluku` |
| Salah ketik | `Dadik Wahyu Cang` / `Chang` |
| Akronim | `Menpora` / `Menteri Pemuda Dan Olahraga`, `BPS` / `Badan Pusat Statistik` |
| Tipe beda | `Deputi 5 (…)` sebagai role / `Deputi 5 ICCN - …` sebagai organization |

Pola-pola ini berbentuk sama dengan hard negative (`Ekonomi Kreatif` ≠ `Ekosistem Ekonomi Kreatif`, `WCCE` ≠ `WCCE 2026`). Karena itu aturan tidak boleh memutuskannya: aturan hanya **mengusulkan**, lalu LLM yang membaca deskripsi kedua sisi **memutuskan**.

## 2. Alur

```
EntityResolver.resolve(..., use_llm=True)           ← satu-satunya titik panggil
  ├─ Fase 1: rencana cluster (alias YAML, kunci, Long (ABBR))
  ├─ Fase 2: EntityAdjudicator.adjudicate()          ← hanya cluster yang belum dikenal registry
  │    L3 kandidat ─► L4 veto ─► L5 LLM ─► L6 pilih merge (star)
  └─ catat keputusan sekali ─► baca ulang registry ─► rewrite node + edge
```

| Lapisan | Isi |
|---|---|
| **L3 Kandidat** | Leksikal (rapidfuzz, per kelompok tipe): containment (`token_set = 100`, kedua sisi ≥ 2 kata), salah ketik (`token_sort ≥ 90`), akronim (inisial atau silabik). Embedding `nama + deskripsi` (model entity VDB) ≥ 0,80 terhadap entity tersimpan dan antar entity baru. Dibandingkan dengan cluster baru lain di dokumen dan semua entity yang dikenal registry workspace. |
| **L4 Veto** | Tipe tidak kompatibel (lintas tipe hanya `concept`↔`program`, `organization`↔`role`); angka beda (`Pasal 9`/`10`); wilayah beda (`Kota Malang`/`Kabupaten Malang`/`Kota Batu`); awalan lembaga beda (`Badan X`/`Dinas X`); dinas/DPRD/Bappeda/Pemerintah generik vs yang berdaerah; `never_merge` YAML. |
| **L5 LLM** | Model `EXTRACT_LLM_*`, temperature 0, ≤ 40 pasangan per panggilan, structured output (`same`, `confidence`, `preferred`, `reason`). Prompt di [entity_resolution.yaml](../../hybrid_rag/ingestion/knowledge_graph/entity_resolution.yaml): konservatif ("ragu → beda") dan memuat contoh hard negative. |
| **L6 Pilih** | Diterima bila `same`, `confidence ≥ 0,95`, dan id pasangan memang diusulkan. Star: entity yang sudah ada selalu jadi canonical dan tidak pernah digabung pergi; tidak ada rantai A→B→C. Antar dua entity baru, nama yang dipilih LLM (`preferred`) yang dipakai. |

> **Perbaikan 2026-10-18 (Fase 3):** di versi awal Fase 2, arah `preferred` terbalik untuk merge antar dua entity baru, sehingga yang bertahan nama yang *tidak* dipilih LLM. Pasangan yang digabung tidak berubah (precision/recall sama), hanya nama tampilan canonical-nya. Sudah diperbaiki dan ada test-nya (`test_new_vs_new_keeps_the_llm_preferred_name`).

Keputusan disimpan di registry dengan `source='llm'`, `confidence`, dan alasan LLM di `evidence`. Rebuild setelah delete membacanya seperti keputusan aturan.

## 3. Cara pakai

```python
await rag.insert_content_list(
    content_list, file_path=path,
    entity_resolution=True,      # Fase 1 (default True)
    entity_resolution_llm=True,  # Fase 2 (default True); False = aturan saja
    entity_registry="iccn",      # opsional
)
```

- Nilai disimpan di metadata doc_status (`entity_resolution_llm`), sehingga retry/reprocess memakai pengaturan yang sama.
- **LLM gagal** (timeout, 5xx, atau output rusak setelah 1 retry): Fase 2 dilewati untuk dokumen itu, dokumen tetap masuk dengan hasil Fase 1, dan ada warning di log. Tidak ada keputusan LLM yang disimpan.
- Konstanta yang bisa disetel ada di awal modul: `MIN_CONFIDENCE`, `EMBED_MIN_SIMILARITY`, `FUZZY_MIN`, `PAIRS_PER_CLUSTER`, `PAIRS_PER_DOCUMENT`, `PAIRS_PER_CALL`.

## 4. Isolasi workspace

Fase 2 hanya membaca dan menulis data workspace yang sedang di-ingest:
- Registry (`known_entity_aliases`), graph (`get_nodes_batch`), dan pgvector (`entities_nearest`) semuanya difilter `workspace` milik instance storage.
- Resolver dan adjudicator dirakit per `IngestionRAG`, yaitu satu per workspace.

Yang dibagi lintas workspace hanya cache file YAML dan gate konkurensi LLM. Keduanya bukan data entity.

## 5. Hasil benchmark

Diukur dengan replay korpus `data/` (14 file) terhadap golden set 1.994 pasangan, termasuk 6 merge Fase 2 yang dilabel belakangan dengan preseden golden set. Graph dan VDB replay hanya berisi dokumen sebelumnya, supaya tidak "melihat masa depan".

| Konfigurasi | Node | Precision | Recall |
|---|---|---|---|
| Tanpa resolusi | 1.974 | – | 0 |
| Fase 1 (aturan) | 1.911 | 1,000 | 0,373 |
| **Fase 1 + Fase 2** (ambang 0,95) | ±1.865 | **0,991 – 1,000** | **0,60 – 0,68** |
| Fase 1 + Fase 2, ambang 0,85 (awal) | 1.837 | 0,949 | 0,80 |

Catatan pembacaan:
- **Kisaran** muncul karena LLM tidak sepenuhnya deterministik walau temperature 0. Satu merge berbeda di dokumen awal mengubah kandidat di dokumen berikutnya.
- **Ambang 0,95** dipilih dari data: di ambang 0,85, semua salah gabung punya confidence ≤ 0,90.
- **Ambang embedding 0,80** juga dipilih dari data. Dengan 0,70, recall turun ke 0,58 dan precision ke 0,98–0,99, karena kandidat embedding yang lebih banyak mendesak kandidat leksikal keluar dari batas 300 pasangan per dokumen. Model titan memberi kemiripan 0,3–0,6 untuk entity yang berkaitan tetapi berbeda, sehingga jalur embedding memang hanya menyumbang sedikit kandidat; sumber utama recall adalah jalur leksikal.
- **Biaya**: ±24 panggilan LLM untuk 14 dokumen (±610 pasangan dinilai). Dua dokumen besar menyentuh batas 300 pasangan per dokumen.

## 6. Batasan yang diketahui

| Batasan | Dampak | Ditangani di |
|---|---|---|
| Dua dokumen yang di-ingest bersamaan tidak saling melihat entity barunya | Duplikat lintas dokumen paralel bisa tersisa | Fase 3: `consolidate_entities()` ([07](07-fase-3-consolidation.md)) |
| Cluster dari dokumen yang Fase 2-nya dilewati/gagal, atau di-ingest dengan `entity_resolution_llm=False`, tidak diadjudikasi ulang saat ingest | Merge yang terlewat sampai konsolidasi dijalankan | Fase 3: `consolidate_entities()` |
| Merge lintas tipe | Tipe node mengikuti mayoritas record, jadi bisa bergeser | Diterima |
| Batas 300 pasangan per dokumen | Dokumen sangat besar tidak semua kandidatnya dinilai | Konstanta `PAIRS_PER_DOCUMENT` |
| Recall yang tersisa (±0,35) | Sebagian besar kualifier yang dinilai LLM < 0,95, ditambah akronim silabik (`Rindekraf`) dan nama orang tersingkat (`Alan W.` / `Alan Wahyu`) | Alias YAML, atau kalibrasi lanjutan |
| Homonim nama persis | Satu node sejak awal (ID node = nama) | Prompt ekstraksi (nama + daerah) |

## 7. Pengujian

- Unit test: [tests/test_entity_adjudication.py](../../tests/test_entity_adjudication.py), mencakup kandidat, veto, ambang, star, kegagalan LLM, target hantu, rebuild, isolasi workspace, dan metadata.
- Benchmark: `evaluation/entity_dedup_benchmark.py evaluate` (opsi `--no-llm`, `--min-confidence`, `--embed-min`). Keputusan LLM dimemo di `evaluation/results/` agar run ulang murah.
