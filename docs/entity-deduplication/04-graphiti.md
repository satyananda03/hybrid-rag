# Analisis Entity Deduplication — `graphiti` (Zep)

> Versi: clone `graphiti/` di root repo (tanpa `pyproject.toml` di root, jadi nomor versi `graphiti-core` tidak tercantum; `server/` mensyaratkan `graphiti-core>=0.28.2`, `mcp_server/` `>=0.31.1`). Snapshot file per 2026-10-17.
> Lingkup kode (path relatif ke `graphiti/graphiti_core/`):
> - Resolusi node: [utils/maintenance/node_operations.py](../../graphiti/graphiti_core/utils/maintenance/node_operations.py), [utils/maintenance/dedup_helpers.py](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py), [prompts/dedupe_nodes.py](../../graphiti/graphiti_core/prompts/dedupe_nodes.py)
> - Kandidat dari graph: [nodes.py `get_by_names`](../../graphiti/graphiti_core/nodes.py#L635), [search/search_utils.py `node_similarity_search`](../../graphiti/graphiti_core/search/search_utils.py#L667), [graph_queries.py](../../graphiti/graphiti_core/graph_queries.py)
> - Orkestrasi: [graphiti.py `add_episode`](../../graphiti/graphiti_core/graphiti.py#L1332), [utils/bulk_utils.py `dedupe_nodes_bulk`](../../graphiti/graphiti_core/utils/bulk_utils.py#L388)
> - Dampak ke edge: [utils/maintenance/edge_operations.py](../../graphiti/graphiti_core/utils/maintenance/edge_operations.py), [utils/maintenance/dangling_endpoints.py](../../graphiti/graphiti_core/utils/maintenance/dangling_endpoints.py), [prompts/dedupe_edges.py](../../graphiti/graphiti_core/prompts/dedupe_edges.py)
>
> **Metode verifikasi (2026-10-17).** Semua klaim perilaku diuji dengan **kode graphiti asli**:
> - **E0:** test bawaan `tests/utils/maintenance/test_node_operations.py` + `test_bulk_utils.py`: **65 passed** (baseline: kode yang dianalisis memang berjalan sesuai test-nya).
> - **E1:** fungsi deterministik asli (`_resolve_with_similarity`, `_promote_resolved_node`, `_collapse_exact_duplicate_extracted_nodes`) pada 23 pasangan kasus target, ditambah pengukuran ambang Jaccard vs panjang nama (300 sampel acak per panjang).
> - **E2:** `Graphiti.add_episode` **end-to-end** di **Neo4j 5.26 Community** (container uji sekali pakai, sudah dihapus) dan Kuzu 0.11.3 embedded. Hanya LLM dan embedder yang palsu:
>   - LLM berskrip: ekstraksi ditentukan per episode, dan dedup memakai **oracle** yang selalu menjawab benar (skenario terbaik untuk LLM) serta mencatat setiap panggilan.
>   - Embedder deterministik: vektor hash trigram huruf, dengan opsi "alias" untuk mensimulasikan embedder yang tahu sebuah singkatan.
>   - Semua query kandidat, keputusan, penulisan, dan efek ke edge adalah kode graphiti.
> - **E3:** `dedupe_nodes_bulk` (jalur `add_episode_bulk`) di level fungsi.
>
> **Keterbatasan:** tidak memakai model embedding sungguhan. Karena itu, *recall* kandidat untuk singkatan (ICCN ↔ bentuk panjang) ditunjukkan untuk dua kondisi ekstrem (embedder tidak tahu vs tahu), bukan diukur untuk model tertentu. LLM oracle berarti setiap kegagalan yang tercatat **bukan** kesalahan LLM, melainkan kesalahan jalur deterministik atau arsitektur.

---

## 0. Validasi claim di [00 §5](00-ringkasan-dan-rekomendasi.md#5-praktik-industri-yang-relevan)

Claim: *"untuk setiap entity baru, ambil kandidat dari graph yang sudah ada lewat hybrid search (embedding + fulltext), lalu biarkan aturan dan LLM memutuskan sebelum menulis. Graphiti (Zep) memakai pola ini."*

| Bagian claim | Status | Bukti di kode |
|---|---|---|
| Resolusi *online*, saat ingest, **sebelum** node ditulis | ✅ Benar | `add_episode` memanggil `resolve_extracted_nodes` ([L1478](../../graphiti/graphiti_core/graphiti.py#L1478)) sebelum `_process_episode_data` menulis ke graph. |
| Kandidat lewat **hybrid search (embedding + fulltext)** | ❌ **Salah untuk kode saat ini** | Kandidat node = identity property + exact name + **cosine saja** ([`_collect_candidate_nodes` L456](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L456)). Fulltext **tidak** dipakai untuk node. Helper versi paper (`hybrid_node_search` [L1174](../../graphiti/graphiti_core/search/search_utils.py#L1174), `get_relevant_nodes` [L1248](../../graphiti/graphiti_core/search/search_utils.py#L1248)) masih ada, tetapi hanya dipanggil oleh test. Hybrid search (BM25 + cosine, `EDGE_HYBRID_SEARCH_RRF`) dipakai untuk **edge** ([L526](../../graphiti/graphiti_core/utils/maintenance/edge_operations.py#L526)). Deskripsi "cosine + full-text" di paper Zep (arXiv 2501.13956 §2.2.1) sudah usang. |
| "Aturan, lalu LLM" | ✅ Benar | exact → gate entropy → MinHash/Jaccard ≥ 0,9 → LLM untuk sisanya ([L840-891](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L840)). |
| "Pola yang terbukti" | ⚠️ Berlebihan | Polanya benar, tetapi implementasi Graphiti menggabungkan homonim lintas tipe dan nama berbeda satu digit **tanpa LLM** (§3), tidak menyimpan keputusan (§4.3), dan hanya aman jika episode di-ingest **berurutan** ([docstring L1409](../../graphiti/graphiti_core/graphiti.py#L1409), terbukti di S8). |

Koreksi atas jawaban saya sebelumnya (yang membaca kode `main` lewat web):
- **"Identity properties" sebagai sumber kandidat.** Faktanya fitur ini **mati** dari API publik. `add_episode`, `add_episode_bulk`, dan `add_triplet` tidak pernah meneruskan `identity_properties`. Selain itu, tidak ada driver yang mengimplementasikan `node_get_by_attribute_values`: base class hanya `raise NotImplementedError` ([L112](../../graphiti/graphiti_core/driver/graph_operations/graph_operations.py#L112)), dan Neo4j memakai `graph_operations_interface = None`.
- **"Cosine ≥ 0,6".** Di Neo4j, `vector.similarity.cosine` mengembalikan **(1 + cos)/2**. Ambang 0,6 setara dengan cosine mentah **≥ 0,2** (terverifikasi S13).
- Deskripsi MinHash/LSH, entropy, Jaccard 0,9, dan LLM fallback **benar**.

---

## 1. Gambaran besar

### 1.1 Posisi dedup dalam `add_episode`

```
extract_nodes (LLM) ─► collapse exact intra-ekstraksi ─► resolve_extracted_nodes ─┐
                                                        (kandidat dari graph,     │
                                                         aturan, LLM batch)       ▼
extract_edges (LLM) ─► materialisasi endpoint yang tidak dikenal ─► resolve lagi ─► resolve_edge_pointers
        ─► resolve_extracted_edges (hybrid search edge, dedup/kontradiksi via LLM)
        ─► extract_attributes + summaries (LLM) ─► tulis (MERGE by uuid)
```

- Paradigmanya **"putuskan dulu, baru tulis"** (*resolve-before-write*). Node yang di-resolve ke kandidat lama memakai **uuid lama**, sehingga penulisan `MERGE (n:Entity {uuid}) SET n = $entity_data` ([node_db_queries.py L183](../../graphiti/graphiti_core/models/nodes/node_db_queries.py#L183)) memperbarui node lama. Tidak ada node duplikat yang ditulis lalu digabung belakangan.
- Lingkup kandidat adalah **`group_id`** (partisi graph). Tidak ada dedup lintas group.

### 1.2 Apa yang *tidak* ada

- **Tidak ada pass dedup offline atau berkala** atas graph yang sudah ada. Prompt `dedupe_nodes.node_list` (pengelompokan massal) ada di [prompts/dedupe_nodes.py L183](../../graphiti/graphiti_core/prompts/dedupe_nodes.py#L183), tetapi tidak dipanggil pipeline mana pun.
- **Tidak ada `merge_entities` atau unmerge** untuk user.
- **Tidak ada penyimpanan keputusan.** `duplicate_pairs` dikembalikan lalu dibuang (`nodes, uuid_map, _ = ...`). Edge `IS_DUPLICATE_OF` tidak pernah dibuat, karena `filter_existing_duplicate_of_edges` ([L1088](../../graphiti/graphiti_core/utils/maintenance/edge_operations.py#L1088)) hanya dipanggil test (S9: 0 edge). Tidak ada alias registry.

---

## 2. Lapisan dedup, satu per satu

### 2.1 L0: prompt ekstraksi dan collapse intra-ekstraksi

- Prompt ekstraksi meminta *"use full names when available"* dan *"avoiding abbreviations"* ([extract_nodes.py L168, L255, L320](../../graphiti/graphiti_core/prompts/extract_nodes.py#L168)). Ini hanya anjuran, bukan jaminan.
- [`_collapse_exact_duplicate_extracted_nodes` L385](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L385) menyatukan nama yang **sama setelah `lower()` + collapse whitespace** dalam satu ekstraksi. Node dengan label lebih spesifik (atau nama lebih panjang) menang.
- **Tidak ada pengecekan tipe.** "Malang" (Location) dan "Malang" (Organization) dalam satu episode menjadi satu node Location, dan Organization hilang tanpa jejak (E1).

### 2.2 L1: pengambilan kandidat — [`_collect_candidate_nodes` L456](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L456)

Tiga sumber kandidat digabung per entity baru:

| Sumber | Implementasi | Catatan |
|---|---|---|
| Identity property | [L487](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L487) | Mati: tidak pernah diberi `identity_properties` oleh API publik, dan lookup driver tidak diimplementasikan (§0). |
| Exact name | [`get_by_names`](../../graphiti/graphiti_core/nodes.py#L635): `MATCH (n:Entity) WHERE n.group_id IN $group_ids AND n.name IN $names` | **Case-sensitive dan whitespace-sensitive** di DB. "kota malang" tidak menemukan "Kota Malang" (S10). Dilewati total jika `group_id` kosong, dan `''` adalah group default Neo4j bila `group_id=None` ([`get_default_group_id`](../../graphiti/graphiti_core/helpers.py#L77), [L549-551](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L549)). |
| Semantik | [`_semantic_candidate_search` L585](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L585): embedding **nama saja**, `node_similarity_search(limit=15, min_score=0.6)` | Neo4j: `MATCH (n:Entity) WHERE group_id … WITH n, vector.similarity.cosine(...)` — **tanpa vector index**. Plan: `NodeIndexSeek(group_id)` lalu cosine untuk *setiap* entity di group (PROFILE). Skala skor (1+cos)/2, jadi 0,6 ≈ cos ≥ 0,2 (S13). Kuzu memakai `array_cosine_similarity` (cos mentah), sehingga ambang yang sama **berarti lain per backend**. |

Konsekuensinya, varian kapitalisasi hanya tertangkap jika embedding nama-nya cukup mirip. Singkatan (ICCN) hanya menjadi kandidat jika embedding nama "ICCN" masuk top-15 terdekat dengan bentuk panjangnya. Tidak ada konteks deskripsi atau chunk yang ikut di-embed.

### 2.3 L2: aturan deterministik — [`_resolve_with_similarity` L321](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L321)

Dijalankan per entity terhadap kandidatnya sendiri:

1. **Exact normalized** (`lower()` + collapse spasi, [L40](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L40)):
   - Tepat 1 kandidat cocok: **merge langsung, tanpa LLM, tanpa cek tipe atau ringkasan**.
   - Lebih dari 1 kandidat cocok: ke LLM (sadar ambiguitas).
2. **Gate entropy** ([L80](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L80)): nama < 6 karakter dan < 2 token, atau entropy karakter < 1,5, dikirim ke LLM. "iccn" (entropy 1,50) lolos gate secara angka, tetapi < 6 karakter dan 1 token, jadi tetap ke LLM.
3. **MinHash/LSH + Jaccard** ([L357-372](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L357)):
   - Nama dinormalisasi oleh [`_normalize_name_for_fuzzy` L46](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L46): selain `[a-z0-9' ]` dibuang, lalu **spasi dihapus** sebelum shingle 3-gram.
   - 32 permutasi, 8 band × 4. Jaccard **≥ 0,9** berarti merge tanpa LLM.
   - **Tidak ada veto** untuk digit, tipe, atau kata pembeda.
   - LSH di sini praktis tidak berguna (kandidat ≤ 15), tetapi tidak merugikan.
4. Sisanya ke LLM.

Ciri penting: lapisan ini **menghapus tanda baca dan spasi** sebelum membandingkan. Akibatnya "Lampiran 1.2" dan "Lampiran 12" menjadi shingle yang identik (J = 1,0).

### 2.4 L3: LLM batch — [`_resolve_with_llm` L649](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L649)

- **Satu panggilan per episode** untuk semua entity yang belum resolved. Kandidat dari semua entity tersebut **digabung** menjadi satu daftar `EXISTING ENTITIES` ([L875](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L875)).
- Konteks yang dikirim:
  - entity baru: nama, label, deskripsi tipe;
  - kandidat: semua atribut, nama, label, **`summary[:120]`** ([L709](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L709));
  - isi episode dan episode sebelumnya.
- Prompt ([`nodes` L118](../../graphiti/graphiti_core/prompts/dedupe_nodes.py#L118)) konservatif: *"Return -1 when there is no match or you are unsure"*. Ada contoh **Java (bahasa pemrograman) ≠ Java (pulau)** ([L170](../../graphiti/graphiti_core/prompts/dedupe_nodes.py#L170)), padahal jalur exact di L2 menggabungkan kasus ini sebelum LLM sempat melihatnya (S2).
- Guardrail respons kuat ([L749-813](../../graphiti/graphiti_core/utils/maintenance/node_operations.py#L749)): id di luar rentang dan id ganda diabaikan, `candidate_id` tak dikenal diperlakukan sebagai "baru", id yang hilang di-log, dan entity tanpa resolusi tetap dibuat sebagai node baru.
- **Field `name` dari LLM tidak dipakai.** Skema `NodeDuplicate.name` meminta *"the most complete and descriptive name"* ([L28](../../graphiti/graphiti_core/prompts/dedupe_nodes.py#L28)), tetapi kode tidak pernah membaca `resolution.name` (S7). Paper menyebut node mendapat "more complete name"; ini tidak lagi berlaku.

### 2.5 L4: survivorship — [`_promote_resolved_node` L171](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L171)

- **Node lama selalu menjadi survivor**, termasuk uuid dan **nama**-nya. Nama baru tidak disimpan sebagai alias.
- Label: hanya dipromosikan bila label lama ⊂ label baru (generic → spesifik). Konflik (Location vs Technology) membuat label lama dipertahankan **tanpa catatan**.
- Atribut ([`_merge_non_empty_attributes` L192](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L192), lalu overlay pada `extract_attributes`):
  - Nilai kosong tidak menghapus nilai lama (bagus).
  - **Nilai baru yang berbeda menimpa nilai lama tanpa jejak** (S3: `kode_wilayah` 3573 → 3577).
- Summary: diregenerasi atau ditambah fakta edge baru oleh LLM.

### 2.6 L5: endpoint edge yang tidak diekstrak sebagai node

- Edge yang endpoint-nya tidak ada di daftar node akan dibuatkan node `['Entity']` lewat [`materialize_dangling_endpoint_node`](../../graphiti/graphiti_core/utils/maintenance/dangling_endpoints.py#L26).
- Node itu lalu dicocokkan dengan node episode berdasarkan exact normalized name, dan sisanya dikirim ke `resolve_extracted_nodes` ([L864](../../graphiti/graphiti_core/graphiti.py#L864)). Ini bagus: endpoint tidak lolos dedup.
- Namun keterbatasan L1–L3 tetap berlaku. Di S11, endpoint "ICCN" menjadi node baru yang terpisah dari "Indonesia Creative Cities Network".

### 2.7 L6: jalur bulk — [`dedupe_nodes_bulk` L388](../../graphiti/graphiti_core/utils/bulk_utils.py#L388)

- **Pass 1:** `resolve_extracted_nodes` per episode secara **paralel** terhadap graph. Episode dalam satu batch tidak saling melihat.
- **Pass 2:** **hanya aturan deterministik** (exact + MinHash) atas gabungan node batch, **tanpa LLM**, lalu `_build_directed_uuid_map` (union-find berarah dengan path compression, [L66](../../graphiti/graphiti_core/utils/bulk_utils.py#L66)).
- Akibatnya, singkatan yang muncul pertama kali di batch yang sama tidak pernah digabung. False merge L2 berlaku sama (E3: Lampiran 1.2/12 dan Java Location/Technology tergabung).

### 2.8 Dampak ke edge

- [`resolve_edge_pointers`](../../graphiti/graphiti_core/utils/bulk_utils.py#L646) memetakan ulang endpoint lewat `uuid_map`, dan membuang edge yang menjadi self-loop karena merge (bagus).
- Dedup edge memakai hybrid search (BM25 + cosine) yang **dibatasi pada pasangan endpoint yang sama**, lalu LLM `dedupe_edges.resolve_edge` memutuskan *duplicate* dan *contradicted* ([L864](../../graphiti/graphiti_core/utils/maintenance/edge_operations.py#L864)).
- Prompt edge justru eksplisit: *"NEVER mark facts as duplicates if they have key differences, particularly around numeric values"* ([dedupe_edges.py L54](../../graphiti/graphiti_core/prompts/dedupe_edges.py#L54)). **Kewaspadaan terhadap angka ini tidak ada di lapisan node.**
- Merge node yang salah ikut **memindahkan fakta** ke node yang salah, lalu fakta milik dua entity berbeda saling dibandingkan sebagai "fakta yang sama" (S12). Karena Graphiti bersifat temporal, jawaban *contradicted* dari LLM akan meng-*invalidate* (`invalid_at`) fakta yang sebenarnya tetap benar.

---

## 3. Uji empiris

### 3.1 E1 — keputusan deterministik `_resolve_with_similarity` (1 kandidat)

| Existing | Baru | J | Keputusan graphiti | Benar? |
|---|---|---:|---|---|
| Indonesia Creative Cities Network | ICCN | 0,00 | LLM (gate entropy) | ke LLM |
| Indonesia Creative Cities Network | Indonesia Creative Cities Networks | 0,97 | MERGE fuzzy | ✅ |
| ICCN | iccn | 1,00 | MERGE exact | ✅ |
| Kota Malang | KOTA MALANG | 1,00 | MERGE exact | ✅ |
| Jawa Timur | Provinsi Jawa Timur | 0,47 | LLM | ke LLM |
| Kementerian Keuangan | Kemenkeu | 0,28 | LLM | ke LLM |
| PT Bank Rakyat Indonesia (Persero) Tbk | PT Bank Rakyat Indonesia Tbk | 0,65 | LLM | ke LLM |
| Badan Perencanaan Pembangunan Nasional | … (Bappenas) | 0,82 | LLM | ke LLM |
| Pasal 1 ayat (2) | Pasal 1 ayat 2 | 1,00 | MERGE fuzzy | ✅ |
| Kota Malang | Kota Madiun | 0,33 | LLM | ✅ |
| Jawa Timur | Jawa Barat | 0,17 | LLM | ✅ |
| Pasal 9 | Pasal 10 | 0,50 | LLM | ✅ |
| UU 24/2007 | UU 25/2007 | 0,33 | LLM | ✅ |
| Kementerian Keuangan | Kementerian Perdagangan | 0,50 | LLM | ✅ |
| LP Cipinang | Cipinang | 0,75 | LLM | ✅ |
| Undang-Undang Nomor 24 Tahun 2007 tentang Penanggulangan Bencana | … Nomor 25 … | 0,88 | LLM | ✅ (nyaris) |
| RPJMD Provinsi Jawa Timur Tahun 2019-2024 (nama lengkap) | … 2024-2029 | 0,899 | LLM | ✅ (nyaris) |
| Dinas Pendidikan **Kabupaten** Malang Provinsi Jawa Timur | … **Kota** Malang … | 0,70 | LLM | ✅ |
| **Lampiran 1.2** | **Lampiran 12** | 1,00 | **MERGE fuzzy** | ❌ false merge |
| **Malang** (Location) | **Malang** (Organization) | 1,00 | **MERGE exact** | ❌ false merge |
| **Java** (Location) | **Java** (Technology) | 1,00 | **MERGE exact** | ❌ false merge |
| **Permendagri Nomor 77** Tahun 2020 tentang Pedoman Teknis … | … **Nomor 78** … | 0,93 | **MERGE fuzzy** | ❌ false merge |
| **Sekolah Menengah Atas Negeri 1** Kota Malang Provinsi Jawa Timur Indonesia | … **Negeri 7** … | 0,90 | **MERGE fuzzy** | ❌ false merge |

**Ambang panjang.** Satu karakter berbeda di tengah nama mengubah 3 shingle di tiap sisi, sehingga J = (n−3)/(n+3) untuk n shingle. Dari 300 nama acak per panjang:

| Panjang (tanpa spasi) | 30 | 40 | 50 | 58 | 60 | 70 | 90 |
|---|---|---|---|---|---|---|---|
| Median J | 0,806 | 0,854 | 0,882 | 0,898 | 0,902 | 0,915 | 0,934 |
| Tergabung otomatis | 0% | 0% | 0% | 0% | **100%** | **100%** | **100%** |

Artinya, **setiap nama ≥ ~60 karakter yang berbeda satu digit (nomor peraturan, nomor sekolah, tahun) digabung tanpa LLM**. Kasus nyata cocok dengan ambang ini: SMAN 1/7 (62 karakter, J = 0,90) dan Permendagri 77/78 (~90 karakter, J = 0,93) tergabung, sedangkan UU 24/25 (~55 karakter, J = 0,88) lolos tipis ke LLM. RPJMD 2019-2024/2024-2029 (J = 0,899) juga lolos tipis, karena lebih dari satu digit yang berbeda.

Survivorship (fungsi asli):
- `_promote_resolved_node`: atribut lama `{kode_wilayah: 3573, luas_km2: 110}` + atribut baru `{kode_wilayah: 3577, luas_km2: ''}` menghasilkan `{kode_wilayah: 3577, luas_km2: 110}`.
- Label Location + Organization menghasilkan `['Entity', 'Location']`.
- Label `Entity` + Location menghasilkan `['Entity', 'Location']`.

### 3.2 E2 — `add_episode` end-to-end (Neo4j 5.26; hasil identik di Kuzu kecuali S11–S13)

| # | Skenario | Hasil di graph | Panggilan LLM dedup | Kesimpulan |
|---|---|---|---:|---|
| S1a | EP1 "Indonesia Creative Cities Network", EP2 "ICCN", EP3 versi UPPERCASE; embedder **tidak** tahu singkatan | 2 node: `ICCN`, `Indonesia Creative Cities Network` (UPPERCASE tergabung) | 0 | Singkatan tidak pernah sampai ke LLM, karena tidak menjadi kandidat. |
| S1b | Sama, embedder **tahu** ICCN ≈ bentuk panjang | 1 node | 1 | Jalur LLM bekerja, tetapi recall **sepenuhnya bergantung** pada embedding nama. |
| S2 | "Java" (Location) lalu "Java" (Technology) | 1 node `Java ['Location']` | **0** | False merge lintas tipe, dan LLM tidak ditanya. |
| S3 | "Kota Malang" `kode_wilayah` 3573, lalu 3577 | 1 node, `kode_wilayah = 3577` | 0 | Nilai lama tertimpa tanpa jejak. |
| S4 | Permendagri 77 lalu 78 (nama panjang) | 1 node (nama 77) | **0** | False merge deterministik. |
| S5 | "Lampiran 1.2" lalu "Lampiran 12" | 1 node | **0** | False merge deterministik. |
| S6 | Bentuk panjang + "ICCN" dalam **satu** episode (embedder tahu singkatan) | 2 node | 0 | Entity baru tidak pernah dibandingkan dengan entity baru lain dalam episode yang sama. |
| S7 | EP1 "Bappenas"; EP2, EP3, EP4 "Badan Perencanaan Pembangunan Nasional"; LLM menjawab duplikat + nama lengkap | 1 node bernama **"Bappenas"** | **3** | Nama dari LLM dibuang. Keputusan yang sama dibayar ulang di setiap episode, karena tidak ada memori alias. |
| S8 | Dua `add_episode` **paralel**, sama-sama "Kota Surabaya" | **2 node "Kota Surabaya"** | 0 | Race: keduanya me-resolve sebelum salah satu menulis. |
| S9 | Jejak keputusan setelah merge | Tidak ada edge `IS_DUPLICATE_OF` | 0 | Merge tidak teraudit. |
| S10 | `get_by_names` untuk "Kota Malang", "kota malang", "Kota  Malang" | Hanya bentuk persis yang ketemu | – | Lookup exact di DB peka huruf dan spasi. |
| S11 | Edge EP2 "ICCN → Kota Malang" dengan ICCN tidak diekstrak sebagai node | Node `ICCN` baru (materialisasi), terpisah dari bentuk panjang | 0 | Endpoint di-resolve, tetapi dengan keterbatasan L1–L3 yang sama. |
| S12 | EP1 "Lampiran 1.2 memuat rincian belanja pegawai RKA…", EP2 "Lampiran 12 memuat daftar sekolah penerima BOS…" | 1 node "Lampiran 1.2" memegang **kedua** fakta | 0 (node), 1 (edge) | LLM edge diminta membandingkan fakta Lampiran 12 dengan fakta Lampiran 1.2 sebagai pasangan endpoint yang sama. |
| S13 | Skor Neo4j vs cosine mentah (embedder trigram) | `neo4j = (1 + raw)/2` di semua baris (mis. 0,717 = (1 + 0,434)/2) | – | Ambang 0,6 di Neo4j = cosine mentah 0,2. |

### 3.3 E3 — `dedupe_nodes_bulk` (3 episode dalam satu batch, graph kosong)

| Canonical | Anggota |
|---|---|
| Lampiran 1.2 | Lampiran 1.2, **Lampiran 12** ❌ |
| Java | Java [Location], **Java [Technology]** ❌ |
| Indonesia Creative Cities Network | … Network, … Network**s** ✅ |
| Kota Malang | Kota Malang, KOTA MALANG ✅ |
| ICCN | ICCN (tidak tergabung; pass 2 tanpa LLM) |
| Kota Madiun | Kota Madiun ✅ |

Panggilan LLM dedup: 0.

---

## 4. Temuan (berdasarkan tingkat keparahan)

### 4.1 [TINGGI] Exact-name merge tanpa cek tipe maupun konteks
Satu kandidat dengan nama yang sama setelah `lower()` langsung digabung, di empat tempat: L2 ([L337-344](../../graphiti/graphiti_core/utils/maintenance/dedup_helpers.py#L337)), collapse intra-ekstraksi, materialisasi endpoint, dan pass 2 bulk. Ini bertentangan dengan contoh "Java" di prompt graphiti sendiri, dan relevan untuk homonim Indonesia: Malang kota/kabupaten, nama orang, nama program yang sama di instansi berbeda. Terbukti di S2, E1, dan E3.

### 4.2 [TINGGI] Fuzzy Jaccard ≥ 0,9 tanpa veto angka atau tanda baca
Normalisasi membuang tanda baca dan spasi. Akibatnya:
- "Lampiran 1.2" ≡ "Lampiran 12".
- Nama ≥ ~60 karakter yang berbeda satu digit digabung otomatis (Permendagri 77/78, SMAN 1/7).

Untuk korpus regulasi dan administrasi, ini kelas kesalahan yang paling sering dan paling merusak: fakta dua peraturan menyatu. Lapisan edge punya peringatan angka, lapisan node tidak.

### 4.3 [TINGGI] Tidak ada persistensi keputusan, alias, atau audit
- Nama dari LLM dibuang.
- Nama varian tidak disimpan sebagai alias.
- `duplicate_pairs` dibuang, dan `IS_DUPLICATE_OF` tidak dibuat.
- Tidak ada unmerge.

Keputusan LLM yang sama dibayar ulang di setiap episode (S7: 3 kali untuk pasangan yang sama), dan merge yang salah tidak bisa ditelusuri.

### 4.4 [TINGGI] Duplikat dalam satu episode tidak pernah di-resolve
LLM hanya membandingkan entity baru dengan `EXISTING ENTITIES` dari DB. Bentuk panjang dan singkatannya yang didefinisikan dalam dokumen yang sama, kasus paling umum untuk ICCN, menghasilkan dua node (S6). Jalur bulk pun hanya memakai aturan deterministik untuk lintas episode dalam batch.

### 4.5 [TINGGI] Tidak aman untuk ingest paralel
Tidak ada lock atau unique constraint pada identitas entity. Dua episode paralel membuat dua node yang sama (S8). Graphiti mendokumentasikan bahwa episode harus di-ingest berurutan ([L1409](../../graphiti/graphiti_core/graphiti.py#L1409)). Duplikat yang terlanjur dibuat tidak akan pernah dibersihkan, karena tidak ada pass offline (§1.2).

### 4.6 [SEDANG] Recall kandidat bergantung pada embedding nama saja
Kandidat semantik meng-embed **nama tanpa konteks**, dan hanya 15 teratas yang diambil. Singkatan, sinonim, dan bentuk dengan prefiks ("Provinsi …") hanya sampai ke LLM bila embedder kebetulan menempatkannya dekat (S1a vs S1b). Ambang `0.6` juga berbeda arti antar backend: Neo4j dan FalkorDB memakai (1+cos)/2 (≈ cos 0,2), Kuzu memakai cos mentah.

### 4.7 [SEDANG] Lookup exact-name di DB rapuh
Lookup ini case-sensitive dan whitespace-sensitive (S10), dan **nonaktif** untuk group default `''` (`group_id=None` di Neo4j). Varian kapitalisasi akhirnya bergantung pada pencarian semantik.

### 4.8 [SEDANG] Konflik nilai ditimpa diam-diam
Atribut terbaru menang dan konflik label diabaikan, tanpa properti audit (S3). Untuk atribut identitas seperti `kode_wilayah`, ini juga menghapus bukti bahwa merge-nya salah.

### 4.9 [SEDANG] Merge node yang salah merambat ke edge temporal
Fakta milik entity berbeda menempel pada node yang sama dan masuk ke dedup/kontradiksi edge sebagai pasangan endpoint yang sama (S12). Jika LLM menjawab *contradicted*, fakta yang benar diberi `invalid_at`.

### 4.10 [SEDANG] `identity_properties` hanya setengah jadi
Logika pencocokannya ada dan diuji, tetapi tidak bisa diaktifkan lewat API publik, dan lookup DB-nya tidak diimplementasikan di driver mana pun (§0). Ini satu-satunya mekanisme berbasis atribut (mis. kode wilayah) yang bisa menjadi pembeda homonim.

### 4.11 [SEDANG] Biaya kandidat semantik O(N) per entity
Tidak ada vector index untuk `name_embedding`, jadi setiap entity baru menghitung cosine terhadap seluruh entity di group-nya (PROFILE: `NodeIndexSeek(group_id)` → `Projection(cosine)` → `Top`). Nama juga di-embed dua kali per episode: sekali untuk mencari kandidat, sekali saat disimpan.

### 4.12 [RENDAH]
- Normalisasi fuzzy membuang karakter non-ASCII (é, aksara non-Latin), sehingga entropy menjadi 0 dan nama dikirim ke LLM. Aman untuk bahasa Indonesia.
- Pemilihan kandidat terbaik saat skor seri bergantung pada urutan iterasi `set`.
- LLM hanya melihat `summary[:120]` dari kandidat.
- Kuzu: `add_episode` dengan `group_id` eksplisit gagal (`KuzuDriver` tidak punya `_database`). Ini bug di luar dedup, dan backend Kuzu sendiri sudah *deprecated*.

---

## 5. Kekuatan (layak diadopsi)

- **Resolve-before-write** dengan uuid survivor. Tidak ada node duplikat yang ditulis lalu harus di-merge di DB, dan tidak perlu `apoc.refactor.mergeNodes`.
- **Deterministik dulu, LLM sebagai fallback**, dengan satu panggilan LLM per episode untuk semua entity yang belum resolved.
- **Sadar ambiguitas:**
  - lebih dari 1 kandidat bernama sama → LLM;
  - nama pendek atau berentropi rendah → LLM, bukan fuzzy.
- **Prompt LLM konservatif** ("unsure → -1") dengan konteks episode, plus **guardrail respons** yang membuat ingest tetap deterministik walaupun LLM salah format.
- **Promosi label generic → spesifik** dan **nilai kosong tidak menghapus nilai lama**.
- **Endpoint edge yang tidak dikenal ikut di-resolve**, dan edge yang berubah menjadi self-loop karena merge dibuang.
- **Union-find berarah** untuk menutup rantai alias di bulk.
- **Kewaspadaan angka di prompt edge.** Prinsip yang sama layak dibawa ke lapisan node.

---

## 6. Penilaian

Graphiti dirancang untuk **memori percakapan agent**: nama orang dan benda pendek, satu pengguna, episode berurutan. Untuk kasus itu desainnya masuk akal: murah, cepat, dan konservatif di LLM. Robustness-nya ada di **lapisan LLM**, bukan di lapisan aturan.

Untuk korpus dokumen Indonesia di hybrid_rag, lapisan aturannya justru menjadi sumber false merge. Penyebabnya:
- exact-name tanpa tipe;
- Jaccard yang buta angka dan tanda baca.

Recall-nya juga lemah pada kasus utama:
- singkatan yang didefinisikan dalam dokumen yang sama;
- varian yang tidak dekat secara embedding nama.

Tidak adanya memori keputusan membuat biaya LLM tidak turun seiring waktu, dan kesalahan tidak bisa diaudit atau dibatalkan.

**Yang layak diambil untuk hybrid_rag:**
- pola resolve-before-write;
- urutan exact → gate → fuzzy → LLM batch;
- guardrail respons LLM;
- promosi label;
- resolusi endpoint edge;
- prompt "unsure → -1".

**Yang harus ditambahkan, karena tidak dimiliki graphiti:**
- veto berbasis bukti (tipe berbeda, digit atau token pembeda berbeda) pada **semua** jalur merge deterministik, termasuk exact-name;
- dedup intra-dokumen sebelum resolusi ke graph;
- alias registry persisten yang juga menyimpan nama dari LLM;
- jejak audit merge dan nilai yang kalah;
- lookup exact berbasis kunci ternormalisasi yang di-index (bukan `n.name IN $names`);
- kandidat semantik lewat vector index dengan konteks deskripsi, bukan nama saja;
- lock per kunci kanonik untuk ingest paralel.
