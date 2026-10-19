# Entity Deduplication untuk `hybrid_rag` — Ringkasan, Perbandingan & Rekomendasi

Dokumen pendamping:
- [01-recon-graphrag.md](01-recon-graphrag.md) — analisis mendalam recon_graphrag 0.10.0 (sudah direview ulang dua kali, langsung ke kode)
- [02-docling-graph.md](02-docling-graph.md) — analisis mendalam docling-graph 1.9.1 (sudah direview ulang dua kali, langsung ke kode)
- [03-neo4j-graphrag-python.md](03-neo4j-graphrag-python.md) — analisis mendalam neo4j-graphrag-python 1.22.0 (sudah direview ulang langsung ke kode dan dijalankan di Neo4j 5.26 + APOC)
- [04-graphiti.md](04-graphiti.md) — analisis graphiti (Zep) sebagai **pembanding pola** *online entity resolution* (dijalankan di Neo4j 5.26). Bukan kandidat untuk di-port.

Target akhir:
1. **Kustomisasi**: `ICCN` ≡ `Indonesia Creative Cities Networks (ICCN)` ≡ `Indonesia Creative Cities Network`.
2. **Variasi urutan/bentuk**: `Kota Malang` ≡ `Malang Kota`, dll.
3. Tanpa merusak pasangan yang **mirip tapi berbeda**: `Kota Malang` ≠ `Kabupaten Malang`, `Kota Malang` (lokasi) ≠ `Pemerintah Kota Malang` (organisasi), `… Pasal 9` ≠ `… Pasal 10`.

> **Status per 2026-10-18 — Fase 1 terpasang dan terukur.**
> - Resolver deterministik ada di [entity_resolution.py](../../hybrid_rag/ingestion/knowledge_graph/entity_resolution.py), dipanggil sebelum tulis dari `merge_nodes_and_edges` dan saat rebuild.
> - Diukur pada korpus `data/` (14 file ICCN: 7 PDF, 1 md, 6 xlsx) dengan golden set 1.988 pasangan ([entity_dedup_golden.csv](../../evaluation/datasets/entity_dedup_golden.csv), script [entity_dedup_benchmark.py](../../evaluation/entity_dedup_benchmark.py)): **precision 1,000, recall 0,384**, 0 panggilan LLM. Rinciannya di §6.7.
> - Benchmark mengoreksi beberapa rekomendasi awal: token-bag tidak boleh dipakai untuk nama lembaga (§4 poin 7), kata sambung dinormalisasi (§4 poin 8), `Long (ABBR)` hanya untuk inisial (§6.2 L2), veto tipe dilonggarkan untuk kandidat LLM (§6.2 L4), dan homonim nama persis adalah urusan ekstraksi, bukan resolver (§6.2 "Batas resolver"). Bagian yang direvisi ditandai **[Revisi Fase 1]**.

---

## 1. TL;DR

| Pertanyaan | Jawaban |
|---|---|
| Mana yang paling **robust**? | **docling-graph**, khususnya di **lapisan alias**. Satu fungsi kunci kanonik dipakai konsisten di semua tahap. Di level graph, LLM hanya boleh mengonfirmasi pasangan yang diusulkan aturan deterministik. Veto yang tidak bisa dibatalkan LLM (bahkan oleh manusia): digit signature, co-occurrence per batch/chunk, tetangga bersama dengan label edge sama, base ambigu. Fusi lintas dokumen membatalkan merge yang memicu konflik nilai. Namun lapisan dasarnya (registry exact) menggabungkan **diam-diam**: homonim lintas parent (nilai bentrok dibuang tanpa catatan), entity ber-ID kosong, dan nilai non-Latin pada field ID selain `name`/`title`/`nom` ([02 §4.5](02-docling-graph.md)). Urutan robustness: docling-graph > recon_graphrag > neo4j-graphrag. |
| Mana yang paling **efisien**? | Untuk keputusan berkualitas: **docling-graph** (LLM id-space; rekonsiliasi skeleton di-batch per ≤100 identifier; alias graph satu panggilan per dokumen, tanpa batching). Tahap termurah secara absolut adalah exact resolver **neo4j-graphrag** (satu agregasi Cypher), tetapi resolver similarity-nya O(n²) per label (±8 menit untuk 10 rb entity dengan spaCy, ±2 menit dengan fuzzy) dan setiap set merge melakukan full scan seluruh DB. recon: LLM rescue mengirim semua pasangan yang lolos blocking, terukur **9–17 panggilan LLM per entity per build** (batas atas 20; 1.000 entity ≈ 16 rb panggilan), tanpa cache sehingga diulang setiap build. Lewat pipeline, rescue selalu aktif untuk strategi hybrid. Blocking O(n²), embedding dihitung tapi tidak dipakai. |
| Apakah salah satunya memenuhi target? | **Tidak.** Diuji dengan kode aslinya: <br>• **recon** gagal di semua kasus target kecuali lewat alias dict manual. Itu pun alias bisa **tidak pernah konvergen**: alias dilewati untuk node yang sudah tergabung, dan survivor "nama terpanjang" bisa menghapus nama canonical. Strategi default `normalized` sendiri sudah salah menggabungkan ("Lampiran 1.2"="Lampiran 12", "LP Cipinang"="Cipinang", dan semua nama yang kosong setelah normalisasi seperti "CO"/"LLC"/"Group"). LLM yang selalu benar pun **tidak menambah satu pasangan benar pun** (2/10, sama dengan fuzzy tanpa LLM), karena kasus target sudah gugur di blocking sebelum sampai ke LLM. Veto `conflict_properties` juga tidak berlaku untuk pasangan *rescue* LLM, sehingga Kota Malang dan Kota Madiun tetap ter-merge walaupun kode wilayahnya berbeda ([01 §6.7](01-recon-graphrag.md)). <br>• **docling**: ICCN↔"… (ICCN)" hanya menjadi kandidat. Kandidat itu diveto bila keduanya anggota list yang sama **atau** berbagi relasi ke entity yang sama (mis. keduanya `LOCATED_IN → Bandung`), termasuk bila manusia sudah mengonfirmasi. Token-swap ditangani deterministik hanya pada mode dense + `aggressive` (bukan default). Di fusi lintas dokumen, satu konflik atribut memecah **semua** entity sekelas antar pasangan dokumen itu, sehingga dedup lintas dokumen nyaris berhenti ([02 §6.6](02-docling-graph.md)). <br>• **neo4j-graphrag** menangkap token-swap, tetapi pada threshold yang sama juga menggabungkan Pasal 9/10, UU 24/25, Kota X/Kota Y, dan (fuzzy) Kementerian Keuangan/Perdagangan. Dijalankan di Neo4j, 12 entity lokasi runtuh menjadi 2 node (spaCy) atau 2–3 node (fuzzy, berubah antar run pada input identik). ICCN = 0. |
| Yang paling mudah diadaptasi ke hybrid_rag? | Konsep **recon** (Neo4j, open-schema, ID deterministik + `MERGE` saat ingest, alias dict global/per-tipe, conflict properties, LLM dengan konteks deskripsi), tetapi implementasinya punya bug fundamental. Dari **neo4j-graphrag** (sudah terinstall) yang layak dipakai hanya pola exact-key di Cypher dan guard per label (dengan catatan: node multi-label menembus guard ini, [03 §4](03-neo4j-graphrag-python.md)). |
| Rekomendasi | **Jangan port satu pun.** Bangun *online entity resolution* di hybrid_rag dengan tiga sumber: guard rails dan disiplin docling-graph, fitur kustomisasi recon, ditambah bagian yang tidak dimiliki ketiganya (alias registry persisten, token-bag, blocking ANN/fulltext terhadap graph yang sudah ada). |

---

## 2. Kondisi hybrid_rag saat ini

### 2.1 Identitas = string nama persis
- Semua nama entity & endpoint relasi lewat satu titik: [`_clean_identifier`](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L427) → sanitize, satukan glyph dash, lalu **`.title()`**.
- Nama itu langsung menjadi `entity_id` (unique constraint `(workspace, entity_id)` di Neo4j), kunci `entity_chunks` KV, dan id VDB `md5(name)` ([L1941](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L1941)).
- Pengelompokan di [`merge_nodes_and_edges`](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L2596) murni exact-match atas string itu.
- Modul resolver belum ada. **[Revisi Fase 1]** Sekarang ada di [entity_resolution.py](../../hybrid_rag/ingestion/knowledge_graph/entity_resolution.py) (menggantikan `semantic_deduplication.py` yang kosong).
- Upaya dedup saat ini hanya di prompt ([knowledge_graph.yaml:10-12](../../hybrid_rag/ingestion/knowledge_graph/knowledge_graph.yaml#L10)), yang hanya berlaku **dalam satu chunk**.

### 2.2 Bug yang harus diperbaiki duluan: `.title()` merusak akronim

> **[Revisi Fase 1] Sudah diperbaiki.** `_clean_identifier` kini memakai `_title_keep_acronyms`: kata huruf kecil di-title-case, kata ALL-CAPS ≥ 2 huruf dipertahankan. Akibatnya `KOTA MALANG` dan `Kota Malang` tidak lagi sama secara string. Penyatuannya sekarang tugas `entity_key`/`match_key` di resolver, sesuai pemisahan display name vs identity key di bawah.
Diverifikasi:

| Output LLM | Disimpan sebagai |
|---|---|
| `ICCN` | `Iccn` |
| `Indonesia Creative Cities Networks (ICCN)` | `Indonesia Creative Cities Networks (Iccn)` |
| `UU No. 24 Tahun 2019` | `Uu No. 24 Tahun 2019` |
| `PT. Telkom Indonesia` | `Pt. Telkom Indonesia` |

Ini bertentangan langsung dengan prompt ("acronyms MUST remain in ALL CAPS"). `.title()` berguna sebagai *case-folding* agar "kota malang" dan "Kota Malang" jadi satu node, tetapi itu tugas **identity key**, bukan **display name**. Akar masalahnya: hybrid_rag memakai satu string untuk dua peran.

### 2.3 Konsekuensi arsitektur: merge pasca-tulis itu mahal
Mengganti nama atau menggabungkan entity yang sudah tersimpan harus menyentuh: node + edge Neo4j, id entity VDB (`md5(name)`), id relation VDB (`md5(src+tgt)`), `entity_chunks` KV, `relation_chunks` KV (kunci pasangan), `full_entities`/`full_relations` per dokumen. Tidak ada utilitas `merge_entities` di codebase. **Karena itu resolusi harus terjadi *sebelum* tulis (saat ingest)**, dengan job offline hanya untuk backfill. recon juga melakukan dedup exact saat ingest (UUID dari ID buatan LLM + `MERGE`), tetapi semua varian diserahkan ke resolusi post-hoc atas seluruh graph di setiap build. Akibatnya alias yang sudah di-merge dibuat ulang sebagai node baru setiap kali dokumen baru menyebutnya ([01 §2.2](01-recon-graphrag.md)).

---

## 3. Perbandingan berdampingan

| Aspek | recon_graphrag | docling-graph | neo4j-graphrag |
|---|---|---|---|
| Kapan berjalan | Dedup exact saat ingest (`MERGE` pada `uuid5(canonical_key)`) + resolusi post-hoc atas seluruh graph setiap build | Saat konversi per dokumen (L3 hanya contract dense untuk dokumen besar) + fusi multi-graph offline (CLI `merge`) | Post-hoc, seluruh DB, setiap run (resolver terakhir di pipeline) |
| Identitas saat tulis | `canonical_key` = ID `<label>:<normalized-name>` yang dinormalisasi **oleh LLM** (tanpa normalisasi kode) | Hash (kelas, field ID kanonik) dari template, **tanpa scope parent/path**. Duplikat digabung first-non-empty; nilai bentrok dibuang tanpa catatan. | Tidak ada (ID per chunk, writer selalu `CREATE`) |
| Identity key | `casefold`, hapus tanda baca & **spasi**, buang suffix badan usaha Inggris di posisi mana pun (→ "Lampiran 1.2"="12", "LP Cipinang"="Cipinang"). Key kosong tidak dijaga ("CO", "LLC", "Group" → `""` → satu node). Suffix Indonesia (`PT`, `Tbk`) tidak dikenali. | Dua rezim menurut **nama field**. `name`/`title`/`nom`: NFKD, hapus diakritik, artikel, posesif, `UPPER_SNAKE` (token dipertahankan). Field lain: casefold + hanya `[a-z0-9]`, sehingga nilai non-Latin menjadi `""` dan runtuh ke satu node. | Tidak ada (exact). `default_process` di fuzzy. |
| Display vs key terpisah | Tidak (nama terpanjang jadi canonical) | **Ya** (hash dari key; atribut menyimpan display) | Tidak (survivor = node pertama menurut urutan scan fisik DB, tidak terkait isi nama) |
| Candidate generation | O(n²); efektif hanya **huruf pertama sama + panjang mirip**; batas 20 kandidat menurut urutan load ([01 §3.3](01-recon-graphrag.md)) | Substring containment per karakter + guard (O(n²) per kelas per dokumen) | Semua pasangan per label |
| Scorer | `fuzz.ratio` (sensitif urutan) | `token_sort_ratio` (hanya dense + aggressive); containment; LLM | `WRatio` atau spaCy cosine, satu threshold |
| Embedding | Nama saja, per pasangan, **tidak dipakai** | Tidak ada | spaCy `en_core_web_lg` (kata Indonesia OOV, tidak terlihat oleh cosine) |
| Alias buatan user | **Ya** (global/per-label), tapi rapuh | Tidak ada | Tidak ada |
| LLM | 1 call/pasangan termasuk semua *rescue* (9–17 call/entity/build, tanpa cache), profil + deskripsi (pada `safe_defaults` ikut field bookkeeping seperti hash dan timestamp), default audit-only | Id-space saja. L3: batch ≤100, bebas dalam satu path. L4: satu call per graph, dibatasi proposer. | Tidak ada |
| Hard negatives | `conflict_properties` (hanya strategi hybrid). Veto ini **tidak berlaku untuk pasangan rescue LLM**, dan satu konflik memblokir seluruh grup. Tanpa guard numerik. | Digit signature, co-occurrence batch/chunk, tetangga bersama (parent **atau child**) berlabel sama, base ambigu, split konflik lintas dokumen yang **menular ke seluruh kelas** | Hanya label (bisa ditembus node multi-label) |
| Clustering | Star (non-transitif). Anchor dipilih menurut urutan load, sehingga hasil bisa beda antar urutan. | Keep/merge group dari LLM, divalidasi | Transitif (dan bug `_consolidate_sets` yang menyebabkan under-merge bergantung urutan scan DB) |
| Human-in-the-loop | `review_groups` dihasilkan, tetapi lewat pipeline tidak dikembalikan dan tidak disimpan | Decision stub `confirm:true` → replay deterministik (node ID stabil), hanya di CLI `merge`, hanya untuk pasangan containment yang diusulkan ulang, dan tetap bisa diveto guard | Tidak ada |
| Provenance merge | `aliases`, fingerprint deskripsi | Fusi: `merged_from`, `__conflicts__`, varian konflik, laporan split. Konversi: hanya `merged_aliases` + statistik L4. Forensik L3 hanya dengan `--debug`. Konflik di registry tidak dicatat. | Hanya relasi `FROM_CHUNK` yang ikut pindah. Properti di-`discard`. |
| Dedup lintas dokumen | `MERGE` uuid saat ingest + resolusi post-hoc | Exact fold di CLI `merge`; satu konflik scalar memecah semua collision kelas itu. Ekspor Cypher `MERGE … SET` melewati semua guard (last-write-wins). `merged_aliases` tidak dipakai ulang sebagai identitas. | Post-hoc atas seluruh DB |
| ICCN ↔ "… (ICCN)" | Tidak (gugur di filter panjang). Alias dict bisa, tetapi bisa tidak konvergen. | Kandidat → LLM. Diveto bila anggota list yang sama atau berbagi child berlabel sama. Di L3 diveto bila pertama muncul di batch yang sama. | Tidak (WRatio 0,60; spaCy 0,00) |
| ICCN ↔ "… Networks" | Tidak (gugur di filter panjang). Alias dict bisa, tetapi bisa tidak konvergen. | Hanya via LLM (L3 dense / prompt direct), probabilistik | Tidak |
| Kota Malang ↔ Malang Kota | Tidak (huruf pertama beda) | Deterministik hanya dense + aggressive. Selain itu bergantung LLM. | **Ya** (fuzzy 0,95; spaCy 1,00) |
| False merge pada negatif uji | **Ya**: default `normalized` menggabungkan Lampiran 1.2/12, LP Cipinang/Cipinang, dan nama ber-key kosong. UU 24/25 dan Pasal 9/10 masuk review dan tergabung bila auto-merge aktif dan LLM menyetujui. Pasangan rescue (Kota Malang/Madiun, Jawa Timur/Barat) tergabung bila LLM menyetujui, walaupun `conflict_properties` dikonfigurasi. | Tidak di lapisan alias (L2/proposer). Tetapi registry menggabungkan entity ber-ID kosong, homonim bernama sama di parent berbeda, dan nilai non-Latin pada field ID selain `name`. | **Ya, masif** (Pasal, UU, Kota X/Y, Jawa Timur/Barat, Jawa Timur/Provinsi Jawa Timur; fuzzy juga Kementerian Keuangan/Perdagangan) |

---

## 4. Bukti: kemampuan tiap teknik pada data target

Diukur dengan key ter-normalisasi (casefold, tanpa diakritik, tanda baca **diganti spasi**, spasi dipertahankan):

| A | B | token-bag sama | inisial = A | `token_sort` | `token_set` | Jaro-Winkler |
|---|---|---|---|---|---|---|
| ICCN | Indonesia Creative Cities Networks (ICCN) | – | – | 19 | 100 | 0.62 |
| ICCN | Indonesia Creative Cities Networks | – | **✓** | 16 | 16 | 0.50 |
| Kota Malang | Malang Kota | **✓** | – | **100** | 100 | 0.49 |
| Kota Malang | Kabupaten Malang ❌ | – | – | 67 | 71 | 0.77 |
| Kota Malang | Pemerintah Kota Malang ❌ | – | – | 67 | **100** | 0.51 |
| Kemenparekraf | Kementerian Pariwisata Dan Ekonomi Kreatif | – | – | 36 | 36 | 0.65 |
| UMKM | Usaha Mikro, Kecil, Dan Menengah | – | **✓** | 12 | 12 | 0.62 |
| Badan Ekonomi Kreatif | Badan Ekonomi Kreatif Indonesia | – | – | 81 | 100 | 0.94 |
| UU … Pasal 9 | UU … Pasal 10 ❌ | – | – | 91 | **96** | **0.98** |

(❌ = pasangan yang **tidak boleh** digabung.)

Pelajaran yang langsung menentukan desain:
1. **Tidak ada satu skor string atau embedding yang memisahkan positif dan negatif.** `token_set` memberi 100 pada "Kota Malang"/"Pemerintah Kota Malang" dan 96 pada Pasal 9/10. Jaro-Winkler memberi 0.98 pada Pasal 9/10. Threshold tunggal terbukti menghasilkan false merge: dengan resolver neo4j-graphrag (`similarity_threshold=0.8`), 12 entity lokasi berbeda runtuh menjadi 2 node dengan spaCy dan 2–3 node dengan WRatio, dan hasil WRatio berubah antar run pada input identik ([03 §10.5](03-neo4j-graphrag-python.md)).
2. **Token-bag equality** adalah aturan presisi-tinggi yang tepat untuk "Kota Malang"↔"Malang Kota".
3. **Inisial** menangkap ICCN dan UMKM, tetapi **akronim silabik khas Indonesia** (Kemenparekraf, Bekraf, Pemkot, Disparbud) tidak bisa diturunkan secara aturan → butuh alias dictionary atau LLM.
4. **Digit-signature guard wajib** (Pasal 9 vs 10), dan angka harus dibaca dengan pemisahnya dipertahankan: "1.2" (digit `1`,`2`) ≠ "12" (digit `12`). recon menghapus tanda baca dan spasi sehingga "Lampiran 1.2" = "Lampiran 12" langsung di-merge oleh strategi default ([01 §6.2](01-recon-graphrag.md)).
5. **Tipe entity adalah guard** (Location vs Organization memisahkan Kota Malang vs Pemerintah Kota Malang).
6. **Embedding model umum berbahasa Inggris tidak cocok** untuk nama Indonesia. Pada spaCy `en_core_web_lg`, token seperti "Kabupaten", "Kementerian", "Pasal", dan akronim adalah OOV dan tidak terlihat oleh cosine. Jika embedding dipakai, gunakan model multibahasa (seperti embedding yang sudah dipakai entity VDB hybrid_rag), dan hanya untuk *blocking*, bukan keputusan.
7. **[Revisi Fase 1] Token-bag hanya aman untuk toponim, tidak untuk nama lembaga.** `Kota Malang`/`Malang Kota` adalah satu tempat yang ditulis dua cara. Sebaliknya, nama dinas adalah nomenklatur yang ditetapkan Perda tiap daerah: urutan urusan dan awalannya bagian dari identitas. Di benchmark, token-bag menggabungkan `Dinas Kepemudaan, Olahraga, Pariwisata, Dan Ekonomi Kreatif` (Papua Barat Daya, Papua Selatan, Papua Tengah) dengan `Dinas Pariwisata, Ekonomi Kreatif, Kepemudaan Dan Olahraga` (Aceh Jaya, Aceh Utara, Gresik, Gunung Kidul). Itu satu-satunya false merge Fase 1, dan kasus kedua yang serupa (`Dinas Kebudayaan Pariwisata & Ekonomi Kreatif` vs `Dinas Pariwisata, Ekonomi Kreatif, Dan Kebudayaan`) muncul begitu kata sambung dinormalisasi. Nama berawalan lembaga (Dinas, Suku Dinas, Badan, Kementerian, Direktorat, Lembaga, Balai, Biro, Kantor, Dewan, Komite, Pusat, Deputi, …) karenanya memakai kunci **berurutan**. Ini juga menjaga `Dinas X` ≠ `Suku Dinas X` ≠ `Badan X` ≠ `X`.
8. **[Revisi Fase 1] Kata sambung dan format penomoran tidak mengubah identitas.** `dan`/`&`, `no`/`nomor`, `tahun` (serta `the`/`of`) dibuang dari kunci pencocokan, tanpa menyentuh awalan lembaga dan urutan angka. Di benchmark ini menambah 11 pasangan benar tanpa false merge: `Badan Riset Dan Inovasi Nasional` = `Badan Riset Inovasi Nasional`, `Perpres No 12 Tahun 2025` = `Perpres Nomor 12 Tahun 2025`, `Perpres No 37/2026` = `Perpres No. 37 Tahun 2026`, `RPJPN 2025-2045` = `RPJPN Tahun 2025-2045`.

---

## 5. Praktik industri yang relevan

Tool entity resolution (Splink/Fellegi–Sunter, Zingg, dedupe.io) umumnya memisahkan tahap **normalisasi → blocking → pairwise scoring → clustering**. Platform MDM (golden record) menambahkan **survivorship, persistensi keputusan, dan review manusia**. Rangkaian lengkap berikut adalah komposit referensi, bukan yang dilakukan setiap tool. Splink, misalnya, berhenti di cluster ID. Senzing bekerja inkremental per record, tanpa tahap batch yang terpisah.

```
normalisasi → blocking (candidate generation) → pairwise scoring → clustering
            → canonicalization/survivorship → persistensi keputusan → review manusia
```

Untuk KG berbasis LLM yang *inkremental* (kasus hybrid_rag), polanya adalah **online entity resolution saat ingest**: untuk setiap entity baru, ambil kandidat dari graph yang sudah ada, jalankan aturan deterministik, lalu serahkan sisanya ke LLM sebelum menulis.

Graphiti (Zep) memakai pola ini, tetapi berbeda dari deskripsi di paper-nya (arXiv 2501.13956):
- Kandidat node di kode saat ini diambil lewat exact name dan cosine embedding **nama saja**, tanpa fulltext. Hybrid search hanya dipakai untuk edge.
- Lapisan aturannya terbukti menggabungkan homonim lintas tipe dan nama yang berbeda satu digit **tanpa LLM**.
- Keputusan merge tidak disimpan.

Graphiti berguna sebagai bukti pola dan sumber guardrail, bukan sebagai implementasi acuan ([04 §0, §4](04-graphiti.md)).

LightRAG/MS GraphRAG (leluhur hybrid_rag) tidak melakukan dedup otomatis selain exact match. LightRAG hanya menyediakan `merge_entities` manual. Itulah gejala yang sekarang Anda alami.

Satu teknik klasik yang tidak dimiliki keempat package dan sangat cocok untuk korpus ini:
- **Alias registry persisten** (gaya MDM / "golden record"): setiap keputusan (dari user, aturan, LLM, atau manusia) disimpan sebagai `alias_key → canonical_id` sehingga **tidak pernah diputuskan dua kali** dan berlaku untuk semua dokumen berikutnya, termasuk saat rebuild.

---

## 6. Rekomendasi desain untuk hybrid_rag

### 6.1 Prinsip
1. **Pisahkan identity key dari display name.** `entity_id` tetap display name (tidak perlu migrasi node lama); identitas dicari lewat `entity_key` + alias registry.
2. **Resolve sebelum tulis**, di satu titik yang dilalui semua jalur (ingest, gleaning, rebuild).
3. **Precision-first** (docling L4): aturan deterministik mengusulkan kandidat, LLM hanya memverifikasi kandidat yang diusulkan, veto struktural tidak bisa dibatalkan. Veto berlaku untuk kandidat berbasis kemiripan, **bukan** untuk merge berbasis bukti (lihat L4). Veto juga harus berdasar bukti *pembeda*, bukan kesamaan: docling memveto dua node yang berbagi relasi ke entity yang sama, padahal relasi yang sama adalah bukti alias.
4. **Satu fungsi kunci kanonik untuk semua jalur** (docling: dipakai di skeleton, registry, proposer, dan re-key, dijaga parity test). Di hybrid_rag artinya ingest, gleaning, rebuild, dan backfill memakai `entity_key()` yang sama, dengan test yang mengunci perilakunya. Normalisasi ada **di kode**, tidak diserahkan ke LLM. recon menyerahkan normalisasi ID ke LLM (`<label>:<normalized-name>`) sehingga konsistensinya tidak terjamin. Perilakunya juga tidak boleh bergantung pada konteks seperti nama field (docling punya dua rezim normalisasi menurut nama field).
5. **Tidak ada merge diam-diam.** Setiap penggabungan, termasuk exact-key, mencatat nilai yang bentrok. Registry docling membuang nilai bentrok tanpa jejak, sementara jalur fusinya mencatat semuanya. Yang kedua yang patut ditiru.
6. **Kustomisasi deklaratif** (recon): alias global/per-tipe, conflict properties, guidance domain.
7. **Setiap keputusan dipersistensi**, termasuk yang ditolak dan diveto → biaya LLM turun seiring korpus tumbuh (positive *dan* negative cache). Alias yang sudah diputuskan harus menjadi **kunci identitas** untuk dokumen berikutnya. Di docling, `merged_aliases` hanya catatan, sehingga bentuk panjang di dokumen baru kembali menjadi node terpisah.
8. **[Revisi Fase 1] Nama lembaga adalah identitas.** Awalan (Dinas, Badan, Kementerian, …) dan urutan kata nama lembaga tidak boleh dilonggarkan oleh aturan mana pun: tidak dibuang, tidak di-token-bag. Nama lembaga yang ditulis berbeda tetapi merujuk lembaga yang sama hanya boleh digabung lewat alias user atau LLM yang melihat deskripsi dan daerahnya (§6.2 L4–L5).

### 6.2 Arsitektur bertingkat

```mermaid
flowchart TD
    A[Nama hasil ekstraksi per dokumen] --> B[L0 Normalisasi: display name + entity_key]
    B --> C{L1 Alias registry hit?}
    C -- ya --> Z[Rewrite ke canonical entity_id]
    C -- tidak --> D[L2 Aturan intra-dokumen: token-bag, 'Long (ABBR)']
    D --> E[L3 Blocking vs graph: alias-key variants + Neo4j fulltext + pgvector top-k, filter tipe]
    E --> F[L4 Fitur + veto: digit sig, tipe, discriminator tokens, conflict props, co-occurrence]
    F -- skor tinggi & lolos veto --> Z
    F -- ambigu --> G[L5 LLM batch per dokumen, profil + deskripsi, hanya pasangan yang diusulkan]
    F -- rendah --> N[Entity baru]
    G -- same & conf ≥ τ --> Z
    G -- tidak --> N
    Z --> R[(Simpan keputusan ke alias registry)]
    N --> R
    R --> M[merge_nodes_and_edges seperti biasa]
```

**L0 — Normalisasi (perbaiki `_clean_identifier`)**
- *Display name*: title-case **hanya** kata yang seluruhnya huruf kecil; pertahankan token ALL-CAPS dan mixed-case (`ICCN`, `UU`, `KaTa`, `PT`). Ini menyelesaikan bug §2.2.
- *entity_key* (untuk lookup, tidak ditampilkan): NFKD + buang diakritik, casefold, satukan dash, buang posesif `'s`/`’s`, **ganti tanda baca dengan spasi (bukan dihapus)**, satukan spasi. Spasi/token dan pemisah angka dipertahankan, sehingga "Lampiran 1.2" → `lampiran 1 2` tetap beda dari "Lampiran 12" → `lampiran 12`. Ini pelajaran dari recon, yang menghapus tanda baca dan spasi lalu langsung merge.
- Varian key turunan untuk blocking: `sorted_tokens_key`, `initialism_key` (abaikan stopword ID/EN: dan, di, ke, dari, untuk, yang, of, and, the), `legal_form_stripped_key` (buang `pt`, `tbk`, `cv`, `persero` **hanya di awal/akhir nama** dan **hanya untuk tipe Organization**). Key ini dipakai **hanya untuk mencari kandidat, tidak pernah untuk auto-merge**. recon membuang kata suffix di posisi mana pun lalu langsung merge, sehingga "LP Cipinang" (Lembaga Pemasyarakatan) = "Cipinang".
- **Key kosong tidak pernah menjadi kunci merge.** Nama yang kosong setelah normalisasi dibuang atau diberi kunci unik. Kedua package mengalami masalah ini: registry docling menggabungkan semua entity ber-ID kosong menjadi satu node ([02 §4.5](02-docling-graph.md)), dan normalisasi recon menggabungkan "CO", "LLC", dan "Group" karena semuanya menjadi `""` ([01 §6.2](01-recon-graphrag.md)).
- **Normalisasi Unicode-aware.** Pertahankan huruf dan angka dari aksara apa pun (`str.isalnum()` / `\w`), jangan hanya `[a-z0-9]`. Di docling, field ID selain `name` memakai `[a-z0-9]`, sehingga "東京", "大阪", dan nama beraksara Jawa semuanya menjadi `""` dan tergabung tanpa peringatan ([02 §6.1](02-docling-graph.md)).
- **[Revisi Fase 1] Yang terpasang.** `entity_key` sesuai rancangan di atas. Kunci pencocokan `match_key` diturunkan darinya:
  - kata sambung (`dan`, `no`, `nomor`, `tahun`, `the`, `of`; `&` sudah hilang sebagai tanda baca) dibuang;
  - nama berawalan lembaga → kunci **berurutan** (`=dinas pariwisata kota malang`);
  - selain itu token-bag: kata diurutkan, token berangka tetap pada urutan aslinya, sehingga `Pasal 9 Ayat 1` ≠ `Pasal 1 Ayat 9`.

  `sorted_tokens_key` dan `legal_form_stripped_key` belum dibuat; `initialism_key` hanya dipakai di aturan `Long (ABBR)`.

**L1 — Alias registry (fitur kustomisasi)**
- Tabel Postgres, misalnya `entity_alias(workspace, alias_key, canonical_entity_id, entity_type, source ∈ {user, rule, llm, human}, confidence, decision ∈ {same, not_same}, evidence, created_at)`.
- Diisi dari: (a) file YAML user, (b) backfill `entity_key` semua node yang ada, (c) keputusan L2–L5.
- Lookup O(1) per nama. Prioritas `user` > `human` > `rule` > `llm`. Entri `not_same` mencegah pasangan yang sama dikirim ulang ke LLM.
- Contoh konfigurasi:

```yaml
# entity_aliases.yaml
global:
  - canonical: "Indonesia Creative Cities Network"
    aliases: ["ICCN", "Indonesia Creative Cities Networks", "Jejaring Kota Kreatif Indonesia"]
  - canonical: "Kementerian Pariwisata dan Ekonomi Kreatif"
    aliases: ["Kemenparekraf", "Kemenparekraf/Baparekraf"]
by_type:
  Location:
    - canonical: "Kota Malang"
      aliases: ["Malang Kota", "Kotamadya Malang"]
never_merge:            # hard negatives yang ditentukan user
  - ["Kota Malang", "Kabupaten Malang"]
discriminator_tokens:   # token yang mengubah identitas bila beda
  Location: [kota, kabupaten, provinsi, kecamatan, desa, kelurahan]
  Program: [plus, pro, lite]
```

Perbedaan dengan alias dict recon (yang terbukti bisa tidak konvergen, [01 §6.3](01-recon-graphrag.md)):
- Canonical **tidak perlu** sudah ada sebagai node; registry yang menentukan id-nya.
- Satu canonical menyerap **semua** aliasnya dalam satu kali proses (recon: satu alias per run).
- Registry dicek **saat ingest**, sebelum node ditulis. recon hanya memakai alias post-hoc, sehingga alias yang sudah di-merge muncul lagi sebagai node baru di dokumen berikutnya.
- **Setiap** alias, termasuk yang tercatat otomatis saat merge, menjadi key registry. recon hanya mencocokkan `name` dan mengabaikan properti `aliases` yang ia simpan sendiri.
- Alias user diproses **sebelum** aturan lain dan tidak bisa dilewati. Di recon, alias dilewati untuk node yang sudah tergabung di tahap normalized/fuzzy.

**[Revisi Fase 1] Desain registry yang terpasang** (dua lapis, keputusan user):
- **YAML per domain**, dipilih **per dokumen** seperti ontology: `insert_content_list(..., entity_registry="iccn")` memuat `knowledge_graph/entity_registry/iccn.yaml`. Nama registry disimpan di metadata doc_status `kg_entity_registry`, sehingga retry/reprocess memakai file yang sama. Isinya `global`, `by_type`, dan `never_merge`. Nama file yang tidak ada langsung error beserta daftar yang tersedia.
- **Tabel Postgres `hybrid_rag_entity_alias` per workspace** untuk keputusan yang sudah diterapkan, termasuk alias YAML yang terpakai. Id baris `<tipe>:<match_key>`. Baris `rule` tidak pernah ditimpa (keputusan pertama menang); baris `user` menimpa. Setelah menulis, resolver **membaca ulang** registry dan memakai nilai tersimpan, sehingga dua dokumen yang di-ingest paralel tetap konvergen ke nama yang sama.
- Prioritas `user > rule` terpasang; `llm`/`human` dan entri `not_same` menunggu Fase 2 (kolom `decision`/`confidence` sudah ada).
- `discriminator_tokens` di contoh YAML di atas **belum** dibaca. Di Fase 1 tidak diperlukan, karena kesamaan `match_key` sudah mengharuskan semua token sama; ini baru relevan untuk kandidat similarity di Fase 2.

**L2 — Aturan deterministik intra-dokumen (gratis, presisi tinggi)**
- *Token-bag equality* pada tipe yang sama → merge (`Kota Malang` ↔ `Malang Kota`).
- *Pola nama `Bentuk Panjang (ABBR)`* → daftarkan `ABBR` dan `Bentuk Panjang` sebagai alias satu sama lain. **Kecualikan sufiks tipe multimodal** `(table)`, `(image)`, `(equation)` yang ditulis oleh [base_processor.py:114](../../hybrid_rag/ingestion/multimodal/base_processor.py#L114).
- *Inisial = singkatan* tanpa pola `Long (ABBR)` → kandidat L5, bukan auto-merge.
- **[Revisi Fase 1]**
  - Token-bag **tidak** berlaku untuk nama berawalan lembaga (§4 poin 7).
  - `Long (ABBR)` hanya diterima bila ABBR = inisial bentuk panjang (dengan atau tanpa stopword). Syarat awal "≥ 2 huruf kapital" dibuang karena `Kota Malang (JATIM)` akan menjadikan "JATIM" alias Kota Malang. Akibatnya akronim silabik (`Kemenparekraf`, `Bekraf`, `Rindekraf`, `Menpora`) tidak tertangkap aturan ini dan harus lewat YAML atau L5.
  - ABBR yang menaut ke lebih dari satu bentuk panjang dalam satu dokumen dianggap ambigu dan tidak ditaut. Kunci yang hanya tersirat (`X` dari `X (A)` tanpa `X` sendiri) tetap dicatat agar dokumen berikutnya menemukannya.
  - Canonical dipilih deterministik: hit registry, lalu bentuk terpanjang; pada panjang sama, bukan-ALL-CAPS menang.

**L3 — Blocking terhadap graph yang sudah ada (skalabel, bukan O(n²))**

Gabungan tiga sumber kandidat, top-k kecil (5–10):
1. Lookup varian key di alias registry (sorted-tokens, initialism, legal-form-stripped).
2. Neo4j fulltext index yang **sudah ada** (`hybrid_rag_entity_id_fulltext`, [graph.py:64](../../hybrid_rag/storage/neo4j/graph.py#L64)).
3. pgvector entity VDB yang **sudah ada** (embedding `name\ndescription`), dengan query `name + description` entity baru.

Filter: tipe kompatibel. Kompleksitas O(m·k) per dokumen (m = entity baru), tidak bergantung pada ukuran graph.

**[Revisi Fase 1] Prioritas kandidat menurut data.** Dari 101 pasangan positif yang terlewat oleh aturan Fase 1 (golden set, §6.7):

| Pola | Jumlah | Contoh | Sumber kandidat yang cocok |
|---|---|---|---|
| Kualifier/containment | 61 | `Dewan Pakar` / `Dewan Pakar ICCN`, `Provinsi Sumatera Utara` / `Sumatera Utara`, `Open Investment` / `Skema Open Investment`, `Sinergi Ekonomi Kreatif` / `Program SINERGI EKONOMI KREATIF` | token containment (setelah membuang kata sambung), fulltext |
| Varian tulis/salah ketik | 29 | `Dadik Wahyu Cang` / `Chang`, `Ahmad Hafidz` / `Hafizh`, `Creative Center` / `Centre`, `Ekosistem 1: Pengembangan Riset` / `Ekosistem Pengembangan Riset` | fuzzy (`token_sort`, edit distance) |
| Akronim silabik | 5 | `Rindekraf`, `Menpora`, `RPJPN` | subsequence huruf + embedding |
| Inisial tanpa pola `(X)` | 4 | `BPS`, `CSR` | `initialism_key` |
| Lain-lain | 2 | tipe beda dengan kunci sama (`ASTA EKONOMI KREATIF`), `Ciptaruang (Placemaking)` | lintas tipe (L4) |

Pola kualifier adalah yang terbesar, **dan** di pola yang sama ada hard negative: `Ekonomi Kreatif` ≠ `Ekosistem Ekonomi Kreatif`, `WCCE` ≠ `WCCE 2026`, `Kota Kreatif` ≠ `Kota/Kabupaten Kreatif`. Jadi containment hanya boleh menjadi **kandidat** L5, tidak pernah auto-merge.

**L4 — Fitur & veto (docling + recon)**
- Veto keras (tidak bisa dibatalkan LLM, berlaku untuk **semua** jalur kecuali alias user): digit signature beda; tipe beda; *discriminator token* beda (kota vs kabupaten); `never_merge`; *conflict properties* ontology (mis. `articleNumber`, tahun regulasi). Veto dievaluasi **tepat sebelum setiap merge dieksekusi**, bukan di satu tahap di tengah pipeline, **termasuk jalur exact-name**. Graphiti menggabungkan "Java" (Location) dengan "Java" (Technology), serta "Lampiran 1.2" dengan "Lampiran 12", di jalur deterministik sebelum LLM sempat melihatnya ([04 §4.1–4.2](04-graphiti.md)). Di recon, veto dijalankan sebelum pasangan rescue LLM ditambahkan, dan auto-merge tidak mengecek ulang, sehingga pasangan yang jelas berkonflik tetap tergabung ([01 §4.4](01-recon-graphrag.md)).
- **Sinyal enumerasi**: dua nama yang diekstrak sebagai entity **terpisah dari chunk yang sama**. Ini *negative feature* untuk kandidat berbasis kemiripan (fuzzy/embedding/LLM), dan **bukan** veto untuk merge berbasis bukti. Alasannya, review docling menunjukkan veto co-occurrence/sibling yang diterapkan mentah memblokir alias yang sah. Definisi "Indonesia Creative Cities Network (ICCN)" dan pemakaian "ICCN" biasanya muncul berdekatan, dan veto docling memblokir ICCN walaupun LLM menyetujuinya, baik di L3 (batch yang sama) maupun di L4 (list yang sama) ([02 §6.3, §6.7](02-docling-graph.md)).
- **Relasi bersama = sinyal positif.** Dua kandidat yang sama-sama terhubung ke entity yang sama dengan relasi yang sama (mis. `LOCATED_IN → Bandung`) menambah skor, bukan memicu veto. docling memveto pola ini, termasuk untuk keputusan manusia ([02 §2.7](02-docling-graph.md)).
- **Conflict properties harus dideklarasikan per tipe**, dan konflik hanya memveto pasangan yang bersangkutan. Jangan memperlakukan *setiap* atribut scalar sebagai pembeda, dan jangan menularkan satu konflik ke seluruh tipe. Di docling, satu beda `role` antar dua dokumen memecah semua Organization di kedua dokumen itu, termasuk yang identik ([02 §6.6](02-docling-graph.md)). Di recon, satu konflik dalam grup memblokir seluruh grup, termasuk anggota yang cocok ([01 §6.7](01-recon-graphrag.md)). Atribut hasil LLM (deskripsi, peran, sektor) sering berbeda antar dokumen dan bukan bukti identitas.
- Auto-merge: key sama / token-bag sama / alias user / pola nama `Long (ABBR)`.
- Kandidat LLM: kombinasi skor (`token_sort`, `token_set`, Jaro-Winkler, cosine embedding, inisial cocok tanpa bukti, containment unik). Mulai dengan aturan eksplisit + threshold; jika data berlabel sudah cukup, ganti dengan bobot Fellegi–Sunter/logistic regression.
- **[Revisi Fase 1] Tipe beda: bukan veto keras untuk kandidat LLM.** 10 dari 164 pasangan positif di golden set diberi tipe berbeda oleh LLM ekstraksi (`ASTA EKONOMI KREATIF` concept vs program, `Deputi 4 ICCN` role vs organization, `Inkubasi Hasil Riset` concept vs program). Tipe beda tetap mencegah **auto-merge** (`Kota Malang` lokasi ≠ `Pemerintah Kota Malang` organisasi), tetapi pasangan lintas tipe boleh diajukan ke L5. Tipe yang memang tidak kompatibel (person vs organization, regulation vs location) tetap diveto.
- **[Revisi Fase 1] Veto daerah untuk nama lembaga.** Dua nama lembaga yang memuat penanda wilayah berbeda (`Dinas Pariwisata Kota Malang` vs `… Kota Batu`, `Kota X` vs `Kabupaten X`) diveto sebelum L5. Penanda wilayah termasuk *discriminator token* (kota, kabupaten, provinsi, kecamatan, desa, kelurahan) beserta nama wilayah yang mengikutinya. Nama lembaga yang sama tetapi ditulis berbeda di daerah yang sama (`Dinas Pariwisata Dan Kebudayaan Kota Malang` vs `Dinas Kebudayaan Dan Pariwisata Kota Malang`, atau vs `Disparbud Kota Malang`) diserahkan ke L5. Relasi bersama ke wilayahnya (keduanya → `Kota Malang`) adalah sinyal positif, kebalikan dari veto tetangga bersama docling ([02 §2.7](02-docling-graph.md) langkah 4), yang justru akan memblokir kasus ini.

**L5 — LLM adjudication (batch, terbatas, berkonteks)**
- **Satu panggilan per dokumen**, dipecah per ≤50 pasangan. Pecahan dikelompokkan **per cluster kandidat**, bukan urutan kemunculan, agar kandidat yang saling terkait tidak terpisah antar batch (kelemahan sub-batching docling L3). Structured output (Pydantic, seperti chain yang sudah ada).
- Payload per pasangan: nama, tipe, deskripsi singkat (≤300 karakter) kedua sisi, alasan kandidat. Deskripsi diperlukan untuk homonim — kelemahan docling yang hanya mengirim identifier. Field payload memakai **allowlist** eksplisit. Default `safe_defaults` recon memakai denylist, sehingga hash fingerprint, timestamp, dan status ringkasan ikut terkirim ke LLM ([01 §6.8](01-recon-graphrag.md)).
- LLM hanya boleh memutuskan pasangan yang diusulkan (guard docling); `confidence ≥ τ` (mulai 0.85) → merge; selain itu → entity baru + entri `not_same` (atau masuk antrean review).
- `llm_guidance` domain dari config (gaya recon).
- **[Revisi Fase 1] Acuan implementasi.**
  - Satu panggilan per dokumen dan prompt konservatif "jawab tidak sama bila ragu", ditambah guardrail respons (id di luar daftar diabaikan, entity tanpa jawaban tetap dibuat baru), mengikuti graphiti L3 ([04 §2.4](04-graphiti.md)).
  - Pembatasan "LLM hanya memutuskan pasangan yang diusulkan" mengikuti docling L4 ([02 §2.7](02-docling-graph.md) langkah 3).
  - Profil deskripsi mengikuti recon ([01 §3.4](01-recon-graphrag.md) langkah 6), dengan allowlist.

  Tidak satu pun dari ketiganya menangani kasus dinas di atas sendirian: docling memvetonya, recon menyaringnya di blocking, graphiti mencari kandidat dari embedding nama saja. Hard negative domain yang dikonfirmasi user (Kemenparekraf ≠ Bekraf ≠ Kementerian Ekonomi Kreatif) masuk ke `never_merge` YAML dan ke contoh prompt.

**L6 — Clustering & canonical**
- Star non-transitif (recon): setiap anggota harus cocok langsung dengan canonical; jangan rantai A~B~C. Closure transitif neo4j-graphrag terbukti meruntuhkan 12 entity lokasi menjadi 2–3 node lewat beberapa pasangan perantara ([03 §10.5](03-neo4j-graphrag-python.md)). Urutan pemrosesan harus deterministik (urutkan menurut key). Star recon memilih anchor menurut urutan load database, sehingga data yang sama bisa menghasilkan 1 atau 2 node ([01 §6.5](01-recon-graphrag.md)).
- Survivor dan nilai properti dipilih **deterministik**: alias user > node yang sudah ada > bentuk terpanjang. **Canonical milik user tidak pernah ditimpa** aturan "nama terpanjang"; di recon aturan itu menghapus nama canonical sehingga alias sisanya tidak pernah tergabung. Deskripsi digabung, bukan di-*discard* seperti `apoc.refactor.mergeNodes(properties:'discard')`. Jika merge dieksekusi dengan APOC, urutkan list node secara eksplisit (survivor di posisi pertama). Jangan mengandalkan urutan `collect`, karena urutannya mengikuti scan fisik DB, bukan urutan pembuatan ([03 §5](03-neo4j-graphrag-python.md)). Properti relasi yang konflik juga tidak mengikuti survivor, jadi relasi domain perlu di-merge secara eksplisit.
- Setelah merge, status ringkasan deskripsi (node **dan** relasi yang dipindahkan) di-reset agar deskripsi gabungan diringkas ulang, cukup **sekali**. Di recon status `success` dari node lama ikut terbawa, sehingga deskripsi hasil merge lintas build tetap mentah. Sebaliknya, pada hybrid dalam satu build, deskripsi diringkas dua kali dan hasil pertama terbuang ([01 §4.8](01-recon-graphrag.md)).
- Properti scalar tidak boleh berubah menjadi list akibat merge (recon: `title`, `created` jadi list). Nilai yang bentrok juga tidak boleh dibuang diam-diam (registry docling: first-non-empty wins tanpa catatan). Simpan nilai yang kalah di properti audit, seperti `__conflicts__` pada fusi docling.
- Canonical display: dari alias user jika ada; selain itu bentuk panjang yang sudah ada di graph (stabil; jangan ganti id node lama). Akronim disimpan di properti `aliases`.
- **Masukkan alias ke konten entity VDB dan properti node** agar query "ICCN" tetap menemukan node "Indonesia Creative Cities Network". Embedding "ICCN" saja secara semantik lemah.

**L7 — Audit & human-in-the-loop (docling)**
- Simpan setiap keputusan (sumber, skor, alasan LLM) di registry. Tambahkan view/endpoint di dashboard untuk antrean `review`; konfirmasi manusia ditulis sebagai `source=human` dan berlaku untuk ingest berikutnya. Antrean review **harus dipersistensi**. Di recon `review_groups` hanya berakhir di log DEBUG, sehingga keputusan LLM yang sudah dibayar terbuang dan diulang di setiap build. Di docling, saat konversi hanya statistik L4 yang tersimpan; pasangan yang ditolak atau diveto hanya muncul di log.
- Veto yang membatalkan konfirmasi manusia harus terlihat, dengan alasannya (pola `ignored_alias_decisions` docling). Manusia juga perlu jalur *override* eksplisit (`source=human` mengalahkan veto berbasis kemiripan, tetapi tidak mengalahkan `never_merge`).
- Lineage di node: `merged_from` / `aliases`.

**[Revisi Fase 1] Batas resolver: homonim bernama persis.**
- **Kasusnya.** Dua lembaga berbeda yang diekstrak dengan nama persis sama, misalnya `Dinas Pariwisata` milik Kota Malang dan milik Kota Batu, sudah menjadi **satu node saat ditulis**, karena `entity_id` = nama (unique constraint `(workspace, entity_id)`).
- **Tidak ada lapisan resolver (L0–L5) yang bisa memisahkannya.** Resolver hanya bisa menggabungkan nama yang berbeda, tidak bisa memecah satu nama. Ini kelas masalah yang sama dengan "exact-name merge tanpa konteks" di graphiti ([04 §4.1](04-graphiti.md)) dan registry docling ([02 §4.5](02-docling-graph.md)).
- **Solusinya di hulu: penamaan di prompt ekstraksi.** Lembaga daerah (Dinas, Badan daerah, Bappeda, DPRD, Pemerintah, …) selalu ditulis beserta daerahnya (`Dinas Pariwisata Kota Malang`), diambil dari konteks bagian dokumen bila tidak disebut di chunk. Aturan yang sama diselaraskan di deskripsi field `name` pada schema ekstraksi. Perubahan prompt dikerjakan terpisah (oleh pemilik proyek).
- **Prompt bersifat probabilistik**, jadi Fase 2 menambah jaring pengaman: nama lembaga generik tanpa penanda wilayah ditandai untuk antrean review, bukan digabung.
- **Pengecualian yang sah:** teks yang memang menyebut lembaga lintas daerah sebagai nomenklatur (mis. slide yang mendaftar nama dinas beserta daftar provinsi yang memakainya). Entity seperti itu bermakna "nomenklatur", bukan satu kantor, dan boleh tetap tanpa daerah.

### 6.3 Titik integrasi di kode

**[Revisi Fase 1]** Kolom status ditambahkan; lokasi diperbarui ke kode sekarang.

| Lokasi | Perubahan | Status |
|---|---|---|
| [`_clean_identifier`](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L436) | Ganti `.title()` dengan smart title-case (pertahankan akronim). Tambahkan fungsi `entity_key()` terpisah. | Terpasang (`entity_key`/`match_key` ada di `entity_resolution.py`) |
| [`merge_nodes_and_edges`](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L2618), setelah `all_nodes`/`all_edges` dikumpulkan dan **sebelum** `_merge_storages(...)` | Rewrite kunci node & endpoint edge ke canonical, buang self-loop yang muncul akibat merge, lalu prefetch berjalan atas nama canonical. | Terpasang: `entity_resolver.resolve(all_nodes, all_edges, kg_entity_registry)` ([L2694](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L2694)); `chunk_texts` belum dipakai (baru relevan untuk L5) |
| [`rebuild_knowledge_from_chunks`](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L660) | Terapkan **lookup registry saja** (deterministik, tanpa LLM). Jika tidak, penghapusan dokumen akan menghidupkan kembali nama alias mentah. | Terpasang: `entity_resolver.canonical_names(...)` setelah parse ulang cache ([L819](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L819)) |
| [entity_resolution.py](../../hybrid_rag/ingestion/knowledge_graph/entity_resolution.py) | Tempat modul resolver (`EntityResolver`, `AliasRegistry`, YAML loader). | Terpasang L0–L2 + L6; L3–L5 Fase 2 |
| [`storage/postgres/sql.py`](../../hybrid_rag/storage/postgres/sql.py#L216) | Tabel `hybrid_rag_entity_alias` + upsert batch. | Terpasang, dibuat oleh `make db-init` |
| Parameter ingest | `insert_content_list(..., entity_resolution=True, entity_registry=...)` → metadata doc_status `entity_resolution` dan `kg_entity_registry` → resolver dipakai saat merge hanya untuk dokumen yang mengaktifkannya. Rebuild tidak memetakan ulang nama dari dokumen yang di-ingest dengan `entity_resolution=False`. | Terpasang |
| Entity VDB upsert ([L1963](../../hybrid_rag/ingestion/knowledge_graph/extraction.py#L1963)) | Sertakan `aliases` di konten embedding. | Fase 3 |
| Script offline baru | Backfill: hitung `entity_key` semua node lama → isi registry → jalankan L2–L5 atas graph lama → utilitas `merge_entities` yang memperbarui **semua** store (§2.3). Lookup node per ID memakai `UNWIND $ids AS i MATCH (n) WHERE elementId(n) = i` (seek), bukan `elementId(n) IN $ids` (full scan, [03 §5](03-neo4j-graphrag-python.md)). | Fase 3 |
| [`evaluation/entity_dedup_benchmark.py`](../../evaluation/entity_dedup_benchmark.py) | `ingest` (merekam input/output setiap `resolve()`), `report`, `candidates`, `evaluate` (replay offline per konfigurasi). | Terpasang |

Konkurensi: dua dokumen yang di-ingest bersamaan bisa sama-sama membuat entity baru yang ekuivalen (masing-masing tidak melihat tulisan yang belum di-commit). Ambil keyed lock atas `entity_key` canonical saat menulis registry; sisa duplikat dibersihkan oleh job backfill berkala. **[Revisi Fase 1]** Yang terpasang tanpa lock: `INSERT … ON CONFLICT DO NOTHING` untuk baris `rule`, lalu baca ulang. Postgres yang memutuskan pemenangnya, dan kedua dokumen memakai nama yang sama. Sisa celah: anggota satu grup bisa terpecah bila hanya sebagian kuncinya kalah balapan.

### 6.4 Bagaimana setiap kasus target tertangani

| Kasus | Lapisan yang menangkap | LLM? | Status Fase 1 |
|---|---|---|---|
| `ICCN` ↔ `Indonesia Creative Cities Networks (ICCN)` | L2 pola `Long (ABBR)` | Tidak | Terpasang, teruji |
| `ICCN` ↔ `Indonesia Creative Cities Network(s)` | L1 alias user; atau L5 | Ya, jika tanpa alias (sekali, lalu tersimpan di registry) | Alias YAML terpasang (`iccn.yaml`); L5 Fase 2 |
| `Kemenparekraf` ↔ `Kementerian Pariwisata dan Ekonomi Kreatif` | L1 alias user; atau L3 embedding → L5 | Ya, jika tanpa alias | Alias YAML terpasang; L5 Fase 2 |
| `Kota Malang` ↔ `Malang Kota` ↔ `KOTA MALANG` | L2 token-bag + case-folding | Tidak | Terpasang, teruji |
| `Kota Malang` ≠ `Kabupaten Malang` | L4 veto discriminator token | Tidak | Aman di Fase 1 (token berbeda, kunci tidak sama); veto eksplisit untuk kandidat similarity di Fase 2 |
| `Kota Malang` ≠ `Pemerintah Kota Malang` | L4 veto tipe (Location vs Organization) | Tidak | Terpasang (tipe bagian dari kunci registry) |
| `… Pasal 9` ≠ `… Pasal 10`, `Lampiran 1.2` ≠ `Lampiran 12`, `Pasal 9 Ayat 1` ≠ `Pasal 1 Ayat 9` | L0 angka dipertahankan berurutan (+ L4 digit-signature untuk kandidat similarity) | Tidak | Terpasang, teruji |
| `iccn` (dokumen ke-50) | L1 registry hit O(1) | Tidak | Terpasang, teruji |
| **[Revisi]** `Dinas Pariwisata, Ekonomi Kreatif, Kepemudaan Dan Olahraga` ≠ `Dinas Kepemudaan, Olahraga, Pariwisata, Dan Ekonomi Kreatif` | L0 kunci berurutan untuk nama lembaga | Tidak | Terpasang, teruji (false merge Fase 1 awal) |
| **[Revisi]** `Badan Riset Dan Inovasi Nasional` ↔ `Badan Riset Inovasi Nasional`, `Perpres No 12 Tahun 2025` ↔ `Perpres Nomor 12 Tahun 2025` | L0 kata sambung dibuang | Tidak | Terpasang, teruji |
| **[Revisi]** `Kota Malang (JATIM)` → "JATIM" **bukan** alias | L2 `Long (ABBR)` hanya inisial | Tidak | Terpasang, teruji |
| **[Revisi]** `Kemenparekraf` ≠ `Bekraf` ≠ `Kementerian Ekonomi Kreatif` | L1 `never_merge` + hard negative di golden set | Tidak | Terpasang di `iccn.yaml` |
| **[Revisi]** `Dinas Pariwisata Dan Kebudayaan Kota Malang` ↔ `Dinas Kebudayaan Dan Pariwisata Kota Malang` / `Disparbud Kota Malang` | L3 kandidat → L4 veto daerah lolos → L5 | Ya | Fase 2 |
| **[Revisi]** `Dinas Pariwisata` (Kota Malang) ≠ `Dinas Pariwisata` (Kota Batu) | Tidak ada; penamaan berdaerah di prompt ekstraksi | — | Di luar resolver (lihat "Batas resolver") |
| **[Revisi]** `Dewan Pakar` ↔ `Dewan Pakar ICCN`, tetapi `Ekonomi Kreatif` ≠ `Ekosistem Ekonomi Kreatif` | L3 containment → L5 | Ya | Fase 2 |

### 6.5 Estimasi biaya

- L0–L4: CPU murah, ditambah 1 query fulltext + 1 query vektor per entity baru yang belum ada di registry. Embedding entity sudah dihitung pipeline saat ini, jadi bisa dipakai ulang.
- L5: ≤ 1–2 panggilan LLM per dokumen, dan cenderung **menurun** karena hit-rate registry naik.
- Bandingkan dengan recon (terukur, [01 §6.4](01-recon-graphrag.md)): blocking O(n²) atas seluruh graph (±40 s untuk 4.000 entity satu label) dan 9–17 panggilan LLM per entity di **setiap** build (1.000 entity ≈ 16 rb panggilan), tanpa cache keputusan. Bila dokumen di-ingest satu per satu lewat `build_from_text`, biaya ini dibayar per dokumen.

### 6.6 Roadmap bertahap

| Fase | Isi | Dampak |
|---|---|---|
| **1** (cepat) — **selesai 2026-10-18** | Perbaiki `.title()`; `entity_key`; alias registry + YAML user; L2 token-bag & `Long (ABBR)`; terapkan juga di rebuild. **[Revisi]** Ditambah: kunci berurutan untuk nama lembaga, normalisasi kata sambung, `Long (ABBR)` hanya inisial, registry YAML dipilih per dokumen. | Kedua target utama selesai secara deterministik. Golden set: precision 1,000, recall 0,384 (§6.7). |
| **2** — **selesai 2026-10-18** ([06](06-fase-2-entity-adjudication.md)): kandidat leksikal + embedding nama+deskripsi, veto, LLM ambang 0,95; golden set: precision 0,991–1,000, recall 0,60–0,68 | L3 blocking (fulltext + pgvector) + L4 veto + L5 LLM batch dengan cache keputusan. **[Revisi]** Prioritas kandidat menurut peta recall (§6.2 L3: kualifier/containment 61, varian tulis 29, akronim 9); veto daerah untuk nama lembaga; tipe beda boleh jadi kandidat LLM; tandai nama lembaga generik tanpa daerah untuk review. Auto-merge LLM di belakang flag sampai lolos §6.7. | Recall lintas dokumen untuk alias yang tidak dikonfigurasi |
| **Hulu** (paralel, di luar resolver) | Prompt ekstraksi: lembaga daerah selalu ditulis beserta daerahnya; field `name` di schema diselaraskan. | Mencegah homonim bernama persis (§6.2 "Batas resolver") |
| **3** — **selesai 2026-10-18** ([07](07-fase-3-consolidation.md)): alias ke node + VDB, konsolidasi offline `consolidate_entities()` + `merge_entities` lintas store, cache verdict. Backfill graph lama tidak dikerjakan (database produksi mulai kosong); antrean review ditunda (butuh UI). | Backfill offline + `merge_entities` lintas store; alias ke VDB; antrean review di dashboard | Membersihkan graph lama; HITL |
| **4** | Golden set berlabel → kalibrasi threshold / bobot Fellegi–Sunter. **[Revisi]** Golden set versi awal (1.988 pasangan) sudah ada sejak Fase 1 dan dipakai untuk mengukur setiap perubahan aturan; Fase 4 tinggal memperluas dan memverifikasinya oleh manusia. | Precision/recall terukur |

### 6.7 Evaluasi (wajib sebelum mengaktifkan auto-merge LLM)
- Buat golden set ±200–500 pasangan dari graph nyata:
  - positif: alias, termasuk alias yang **berbagi relasi** dan alias yang muncul di **chunk yang sama** (definisi singkatan);
  - **hard negatives**: Kota/Kabupaten, Kota X/Kota Y dengan kode wilayah berbeda, Pasal n/n+1, penomoran desimal seperti Lampiran 1.2/12, singkatan lembaga seperti LP Cipinang/Cipinang, nama pendek yang bisa ternormalisasi kosong (CO, LP), tier program, Kementerian lama vs baru, kementerian berbeda dengan awalan sama (Kementerian Keuangan/Perdagangan), **homonim bernama persis sama** dengan konteks berbeda (mis. "Dinas Pariwisata" milik dua kota), dan nama non-Latin.
- Metrik: pairwise precision/recall dan B-cubed F1 per tipe. Target awal: **precision ≥ 0.98** (false merge merusak data dan sulit dibalik; duplikat hanya redundan, prinsip docling), lalu naikkan recall.
- Uji regresi: setiap bug false-merge menjadi entri `never_merge` + test case.

**[Revisi Fase 1] Evaluasi yang sudah dijalankan.**

*Korpus dan cara ukur.*
- Korpus: `data/`, 14 file ICCN (6 xlsx jejaring per wilayah, 7 PDF, 1 md Perpres 37/2026). Di-ingest ke workspace `workspace_entity_test` dengan resolusi aktif tanpa YAML. Setiap panggilan `resolve()` merekam nama mentah, tipe, deskripsi, dan mapping-nya.
- **Kandidat dijaring tanpa memakai aturan resolver**: kesamaan kunci (lintas tipe), `token_sort`/`token_set` ≥ 88 (token_set hanya bila kedua nama ≥ 2 token), inisial, akronim silabik (subsequence huruf), pola `(X)`, ditambah semua pasangan yang digabung sistem. Hasilnya 1.974 nama mentah dan **1.988 pasangan**.
- **Label**: 4 agen AI terpisah, *blind* (tanpa keputusan sistem dan sinyal kandidat), memakai pedoman yang sama, termasuk hard negative domain dari pemilik proyek. Pasangan "ragu" dan pasangan yang bertentangan dengan sistem diadjudikasi ulang. Hasil: **164 sama, 1.821 beda, 3 ambigu** (nama orang umum yang hanya beda kapitalisasi, dengan peran berbeda).
- Konfigurasi lain diukur dengan **replay offline** atas nama mentah yang terekam. Replay aturan saat ingest menghasilkan angka yang identik dengan ingest nyata.

| Konfigurasi | Node | TP | FP | FN | Precision | Recall | F1 | Precision terburuk* |
|---|---|---|---|---|---|---|---|---|
| Tanpa resolusi (exact string) | 1.974 | 0 | 0 | 164 | – | 0 | – | – |
| Fase 1 awal (ingest nyata) | 1.919 | 52 | 1 | 112 | 0,981 | 0,317 | 0,479 | 0,929 |
| Fase 1 + guard lembaga + kata sambung | 1.911 | 63 | 0 | 101 | **1,000** | **0,384** | 0,555 | 0,955 |
| Idem + `iccn.yaml` | 1.909 | 65 | 0 | 99 | 1,000 | 0,396 | 0,568 | 0,956 |

\*Precision terburuk: 3 pasangan ambigu dihitung sebagai salah gabung.

**[Revisi Fase 2]** Dengan adjudikasi LLM (ambang confidence 0,95, golden set ditambah 6 merge Fase 2 yang dilabel berdasarkan preseden): precision **0,991–1,000**, recall **0,60–0,68**, ±24 panggilan LLM untuk 14 dokumen. Pada ambang awal 0,85, precision 0,949 (di bawah syarat 0,98); semua salah gabungnya ber-confidence ≤ 0,90. Rincian di [06 §5](06-fase-2-entity-adjudication.md).

*Efisiensi.* 0 panggilan LLM. Total waktu `resolve()` 25 detik untuk 2.101 record entity (±1,8 detik/dokumen, ±12 ms/entity). Waktu itu hampir seluruhnya round-trip Postgres saat pool sedang sibuk menulis hasil ingest; aturannya sendiri hampir tanpa biaya.

*Keterbatasan.*
- Recall relatif terhadap kandidat yang dijaring: alias yang tidak tertangkap sinyal mana pun tidak terhitung.
- Label dibuat AI dan perlu spot-check manusia, terutama pasangan berlabel "sama" dan catatan `unsure` di golden set.
- Korpus ini didominasi daftar nama orang dari xlsx, sehingga variasi alias lembaga lebih sedikit dibanding korpus regulasi.

*Cara menjalankan:*

```bash
.venv/bin/python evaluation/entity_dedup_benchmark.py ingest [--entity-registry iccn]   # sekali, memakai LLM ekstraksi
.venv/bin/python evaluation/entity_dedup_benchmark.py candidates   # kandidat untuk dilabel ulang/ditambah
.venv/bin/python evaluation/entity_dedup_benchmark.py evaluate     # precision/recall + replay per konfigurasi
```

---

## 7. Kesimpulan

- **Paling robust dan efisien (per keputusan berkualitas):** docling-graph. Kekuatannya:
  - satu kunci kanonik yang konsisten;
  - LLM id-space yang dibatasi proposer dan veto;
  - split konflik lintas dokumen;
  - audit fusi yang lengkap dan keputusan manusia yang bisa di-replay.

  Kelemahannya ada di dua ujung:
  - **Terlalu ketat di lapisan alias.** Recall deterministik sempit (containment; token-swap hanya di dense + aggressive). Veto co-occurrence dan tetangga bersama ikut memblokir alias yang sah, termasuk yang dikonfirmasi manusia. Satu konflik atribut menghentikan dedup lintas dokumen untuk seluruh kelas.
  - **Terlalu longgar di lapisan dasar.** Registry exact menggabungkan homonim, ID kosong, dan nilai non-Latin, lalu membuang nilai bentrok tanpa jejak. Ekspor Cypher `MERGE … SET` melewati semua guard.

  Tidak ada alias kustom, dan alias yang sudah diputuskan tidak dipakai ulang sebagai identitas.
- **Paling dekat secara arsitektur dan kustomisasi:** recon_graphrag. Polanya layak diadopsi: ID deterministik + `MERGE` saat ingest, alias global/per-tipe, conflict properties, profil LLM berbasis deskripsi, dan grouping star. Kelemahan implementasinya (semua terverifikasi pada kode aslinya):
  - blocking efektif hanya "huruf pertama + panjang", sehingga semua kasus target gugur sebelum dinilai;
  - LLM (bahkan oracle) tidak menambah recall, karena kandidatnya hanya pasangan yang sudah lolos blocking;
  - strategi default sudah salah menggabungkan (Lampiran 1.2/12, LP Cipinang, nama ber-key kosong);
  - alias bisa tidak pernah konvergen;
  - veto `conflict_properties` bocor di jalur rescue LLM, dan tidak ada guard numerik;
  - hasil bergantung urutan load;
  - 9–17 panggilan LLM per entity per build tanpa cache, dengan blocking O(n²) atas seluruh graph.
- **Paling berisiko:** neo4j-graphrag. Exact resolver-nya aman tetapi tidak menangkap variasi apa pun. Resolver fuzzy/spaCy-nya menangkap token-swap, tetapi menggabungkan entity berbeda secara masif (Pasal, UU, Kota X/Y, Kementerian X/Y), membuang properti tanpa jejak, dan hasil fuzzy-nya bahkan berubah antar run pada input identik.
- **Untuk hybrid_rag:** bangun resolver online sebelum tulis, dengan alias registry persisten sebagai inti kustomisasi, aturan deterministik khas Indonesia (token-bag, singkatan, discriminator token administratif), blocking memakai fulltext index & pgvector yang sudah ada, veto ala docling (hanya yang berbasis bukti pembeda: digit, tipe, discriminator token, conflict properties yang dideklarasikan), dan LLM batch ala docling yang diberi konteks deskripsi ala recon. Mulai dari Fase 1. Fase itu saja sudah menyelesaikan ICCN dan Kota Malang tanpa satu pun panggilan LLM.
- **[Revisi Fase 1] Hasil Fase 1.** Aturan deterministik saja sudah menyelesaikan target ICCN dan Kota Malang, dengan precision 1,000 dan recall 0,384 pada golden set 1.988 pasangan dan tanpa panggilan LLM. Benchmark mengoreksi satu asumsi penting dokumen ini: token-bag aman untuk toponim, tetapi tidak untuk nama lembaga, yang awalan dan urutan katanya adalah identitas. Recall yang tersisa didominasi kualifier/containment, tempat hard negative juga berada, jadi Fase 2 harus memakai LLM berbasis deskripsi untuk pola itu, bukan menambah aturan. Homonim bernama persis tetap di luar jangkauan resolver dan diselesaikan di prompt ekstraksi.
