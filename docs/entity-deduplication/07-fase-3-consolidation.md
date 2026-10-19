# Entity Resolution Fase 3 — Alias di VDB dan Konsolidasi Duplikat Tersimpan

Lanjutan dari [05 (Fase 1)](05-fase-1-entity-resolution.md) dan [06 (Fase 2)](06-fase-2-entity-adjudication.md). Kode: [entity_consolidation.py](../../hybrid_rag/ingestion/knowledge_graph/entity_consolidation.py), alias di [entity_resolution.py](../../hybrid_rag/ingestion/knowledge_graph/entity_resolution.py), dan cache verdict di [entity_adjudication.py](../../hybrid_rag/ingestion/knowledge_graph/entity_adjudication.py).

## 1. Masalah yang diselesaikan

| Masalah | Penyebab | Solusi Fase 3 |
|---|---|---|
| Query pakai singkatan ("ICCN") tidak menemukan node setelah merge | Nama `ICCN` hilang dari teks yang di-embed | **Alias** ikut di node dan di teks entity VDB |
| Duplikat dari dokumen yang di-ingest bersamaan | Saat resolve, kedua dokumen tidak saling melihat | **Konsolidasi** antar entity tersimpan |
| Duplikat dari dokumen yang Fase 2-nya gagal atau dimatikan | Nama yang tidak dinilai LLM tidak pernah dinilai ulang | **Konsolidasi** |

Asumsi proyek: database produksi mulai dari kosong, jadi tidak ada backfill graph lama. Human review ditunda (butuh UI). Verdict zona abu-abu sudah tersimpan untuk fitur itu nanti.

## 2. Alias

- Nama yang digabung resolver dicatat di properti node `aliases` (string `<SEP>`, maksimal 10).
- Teks entity VDB menjadi `"<nama>\nAlias: ICCN, …\n<deskripsi>"`. Alias ditaruh sebelum deskripsi karena pemotongan teks memotong dari belakang.
- Hanya alias yang **kata-katanya berbeda** yang disimpan. `KOTA MALANG` tidak menambah informasi pada `Kota Malang`, sedangkan `ICCN` menambah.
- **Entity tanpa alias tidak berubah sama sekali**, sehingga tidak di-embed ulang.
- **Retrieval tidak diubah.** Konteks LLM hanya mengambil properti tertentu (entity, tipe, deskripsi), jadi `aliases` hanya memengaruhi pencarian lewat embedding.
- **Saat rebuild setelah delete**, alias dihitung ulang dari record yang tersisa, sehingga alias milik dokumen yang dihapus ikut hilang.

Method: `EntityResolver.merged_aliases()` dan `EntityResolver.vdb_content()`, dipanggil dari tiga titik pembentukan teks VDB di `extraction.py`.

## 3. Konsolidasi

```python
from hybrid_rag.ingestion import consolidate_entities

report = await consolidate_entities("ws", dry_run=True)   # rencana merge saja
for m in report.merges:
    print(m.absorbed, "->", m.survivor, m.confidence, m.reason)
report = await consolidate_entities("ws", dry_run=False)  # eksekusi
```

`EntityConsolidator.run()`:
1. **Pengaman**: batal (`RuntimeError`) bila ada dokumen PENDING/PARSING/PROCESSING di workspace itu. Pengecekan dilakukan saat mulai dan lagi sebelum eksekusi, karena kunci graph hanya berlaku di dalam satu proses.
2. Membaca semua entity workspace, kecuali multimodal. Embedding diambil dari entity VDB yang sudah tersimpan, jadi **tanpa biaya embed**.
3. `EntityAdjudicator` yang sama dengan Fase 2: kandidat leksikal + embedding → veto (`never_merge` dari **gabungan semua YAML**) → LLM (ambang 0,95) → star.
4. **Survivor** = entity dengan sumber chunk terbanyak, lalu relasi terbanyak. Bila seri, nama pilihan LLM.
5. **Cache verdict** (`hybrid_rag_entity_pair_verdict`): pasangan yang pernah dinilai tidak ditanyakan lagi. Dry-run juga mengisi cache ini, supaya apply tidak membayar LLM dua kali. Dry-run **tidak** menulis apa pun ke graph, VDB, atau registry.

`EntityConsolidator.merge(survivor, absorbed)`, di bawah keyed lock graph, memakai ulang fungsi yang sudah ada:

| Langkah | Store |
|---|---|
| Node yang diserap dijadikan record → `_merge_nodes_then_upsert(survivor)` | Node survivor (deskripsi, sumber, tipe, alias), entity VDB, `entity_chunks` |
| Setiap relasi diserap → `_merge_edges_then_upsert(survivor, tetangga)`. Tipe ontology dan arahnya dipertahankan; bobot dihitung sekali. | Edge, relation VDB, `relation_chunks` |
| `_remove_entities({absorbed})` (dari purge) | Hapus node lama, baris VDB, dan chunk tracking-nya |
| Nama diserap → survivor | `full_entities` / `full_relations` per dokumen (supaya delete tetap benar) |
| `AliasRegistry.repoint()` | Registry: ingest dan rebuild berikutnya memetakan ke survivor |

## 4. Satu workspace dengan beberapa YAML registry

- YAML dipilih per dokumen. Alias YAML yang terpakai disimpan sebagai keputusan `user`.
- Kalau dua YAML memetakan alias yang sama ke canonical berbeda, YAML yang dipakai **terakhir** berlaku untuk dokumen berikutnya. Node lama tetap di canonical lamanya, dan konsolidasi bisa menyatukan keduanya bila LLM mengonfirmasi.
- Veto `never_merge` di konsolidasi memakai **gabungan** semua YAML. Ini aman untuk precision, karena workspace bisa berisi dokumen dengan registry berbeda.

## 5. Latency

| Jalur | Dampak |
|---|---|
| Ingest | Alias hanya operasi string. Embed ulang hanya terjadi pada node yang alias-nya berubah, dan node itu memang sedang di-upsert. Cache verdict tidak dipakai saat ingest. |
| Konsolidasi | Di luar ingest, jadi tidak menambah latency ingest. Biaya: LLM untuk pasangan **baru** saja, plus satu merge per pasangan. |

## 6. Batasan

| Batasan | Catatan |
|---|---|
| Jangan jalankan saat workspace sedang ingest di proses lain | Ada pengecekan status dokumen; dokumen yang masuk antrean di tengah eksekusi merge tidak terdeteksi (jendelanya kecil) |
| Kandidat embedding antar entity dihitung N×N di memori | Cukup untuk ribuan entity; puluhan ribu perlu blocking |
| Maksimal 2.000 pasangan per run | Run berikutnya melanjutkan, karena pasangan yang sudah dinilai diambil dari cache |
| Survivor yang sudah di batas sumber chunk (mode KEEP) | Sumber dari node yang diserap bisa tidak semuanya tercatat |

## 7. Pengujian
- Unit: [tests/test_entity_consolidation.py](../../tests/test_entity_consolidation.py) — alias, cache verdict, rencana konsolidasi (survivor, `never_merge`, tanpa penulisan), pengaman ingest, record edge ontology, indeks dokumen, isolasi workspace.
- End-to-end: lihat hasil di §8.

## 8. Hasil uji end-to-end

Database `retrieval_toolkit`, workspace `ws_er_consol2`. Tiga PDF (`Overview ICCN`, `10 Prinsip Kota Kreatif`, `11 Jurus Catha Ekadasa`) di-ingest dengan `entity_resolution_llm=False`, supaya duplikat yang biasanya ditangkap Fase 2 tersisa. Hasilnya 77 node dan 192 edge. Semua cek lolos:

| Cek | Hasil |
|---|---|
| Ada dokumen PENDING → konsolidasi batal | ✅ |
| Dry-run: 10 pasangan dinilai, 1 panggilan LLM, 1 rencana merge; graph tidak berubah | ✅ `Musrenbang Interaktif (Design Action)` → `Musrenbang Interaktif` (0,95) |
| Apply mengeksekusi persis rencana dry-run, 0 panggilan LLM baru (10 verdict dari cache) | ✅ |
| Node 77 → 76; node yang diserap hilang dari graph, entity VDB, dan `entity_chunks` | ✅ |
| Tidak ada edge, baris registry, atau indeks dokumen yang masih menunjuk nama yang diserap | ✅ |
| Setiap relasi survivor punya baris relation VDB | ✅ |
| Survivor membawa alias (properti node dan teks VDB) | ✅ 1/1 |
| Query entity VDB dengan nama yang diserap menemukan survivor di peringkat 1 | ✅ |
| Run kedua: 0 merge, 0 panggilan LLM | ✅ |
| Delete dokumen setelah konsolidasi berhasil, dan nama yang diserap tidak hidup kembali | ✅ |
| Workspace lain (`ws_er_a`) identik sebelum/sesudah | ✅ |

**Bug yang ditemukan uji ini dan sudah diperbaiki:** confidence di cache verdict (kolom `REAL`) terbaca `0.9499999` untuk 0,95, sehingga apply menolak merge yang direncanakan dry-run. Sekarang confidence dibulatkan saat dibaca, dan ada test regresinya.
