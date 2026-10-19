# Analisis Entity Deduplication — `recon_graphrag`

> Versi: **recon-graphrag 0.10.0** (`recon_graphrag/pyproject.toml`), berlabel *Work in Progress* di README.
> Lingkup kode (path relatif ke `recon_graphrag/recon_graphrag/`):
> - Ekstraksi & identitas: [extraction/prompts.py](../../recon_graphrag/recon_graphrag/extraction/prompts.py), [extraction/parser.py](../../recon_graphrag/recon_graphrag/extraction/parser.py), [extraction/validator.py](../../recon_graphrag/recon_graphrag/extraction/validator.py), [extraction/extractor.py](../../recon_graphrag/recon_graphrag/extraction/extractor.py), [extraction/assembler.py](../../recon_graphrag/recon_graphrag/extraction/assembler.py), [extraction/structured.py](../../recon_graphrag/recon_graphrag/extraction/structured.py), [pipelines/writer.py](../../recon_graphrag/recon_graphrag/pipelines/writer.py)
> - Resolusi: [graphdb/entity_resolution.py](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py), [graphdb/entity_resolution_context.py](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution_context.py), [graphdb/neo4j/entity_resolution.py](../../recon_graphrag/recon_graphrag/graphdb/neo4j/entity_resolution.py), [graphdb/memgraph/entity_resolution.py](../../recon_graphrag/recon_graphrag/graphdb/memgraph/entity_resolution.py), [graphdb/neo4j/store.py](../../recon_graphrag/recon_graphrag/graphdb/neo4j/store.py)
> - Orkestrasi: [pipelines/graphrag_pipeline.py](../../recon_graphrag/recon_graphrag/pipelines/graphrag_pipeline.py), [extraction/description_summarizer.py](../../recon_graphrag/recon_graphrag/extraction/description_summarizer.py)
>
> **Metode verifikasi (review ulang ke-2):**
> 1. Seluruh modul di atas dibaca ulang baris per baris. Setiap klaim di dokumen ini merujuk ke baris kode.
> 2. Test resmi dijalankan: `tests/graphdb` → **106 lulus**, termasuk **49 test entity resolution**. Test ini butuh `pytest-asyncio` (dev dependency recon) yang tidak terpasang di `.venv` hybrid_rag. Tanpa plugin itu, 42 test async gagal karena lingkungan, bukan karena kode. **Tidak ada test** untuk blocking/akronim, `llm_rescue`, konvergensi alias, atau interaksi `conflict_properties` dengan rescue.
> 3. Kode strategi asli (`BaseEntityResolver`) dijalankan lewat harness in-memory. Hanya `_load_entities`/`_merge_groups` yang diganti store di memori (survivor = nama terpanjang, properti `combine`, `aliases +=`, sama seperti backend). LLM dan embedder dibuat deterministik (oracle / "selalu ya" / "selalu tidak").
> 4. Perilaku khusus Neo4j/APOC **tidak dieksekusi** (tidak ada instance Neo4j yang tersedia). Klaim tentang hal itu ditandai *(pembacaan kode)*.
>
> Hasil eksperimen ada di §6.

## Catatan revisi

### Revisi ke-2 (review ulang ke kode asli)

| # | Klaim sebelumnya | Fakta setelah verifikasi |
|---|---|---|
| R2-1 | `conflict_properties` adalah "hard veto" (§5) dan "satu-satunya pengaman" (§4.4). | **Veto tidak berlaku untuk pasangan `llm_rescue`.** Conflict rules dijalankan *sebelum* rescue ditambahkan ([L552](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L552) vs [L572](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L572)), dan `_promote_ai_reviews` tidak memeriksa konflik. Terverifikasi: Kota Malang (kode 3573) dan Kota Madiun (3577) tetap auto-merge walaupun `conflict_properties={"Location": ["kode_wilayah"]}` (§6.7). |
| R2-2 | (tidak dibahas) | Nama yang ternormalisasi menjadi **string kosong** ("CO", "LLC", "Group", "Co.", "LP") tergabung semua dalam satu label. "CO" (karbon monoksida) lenyap ke node "Group" (§6.2). |
| R2-3 | Profil LLM "menyaring field internal" (§3.4, §5). | Pada `safe_defaults`, profil ikut membawa field bookkeeping: `descriptions` (duplikat `description`), `observation_count`, `type`, `description_summary_status`, `description_input_fingerprint` (hash 64 karakter), dan `description_summary_updated` (timestamp). Ini bertentangan dengan dokumentasi recon sendiri ([docs/05-pipelines.md L263-266](../../recon_graphrag/docs/05-pipelines.md#L263)) (§6.8). |
| R2-4 | (tidak dibahas) | Lewat pipeline, `hybrid` **selalu** menerima `llm` dan `embedder` ([graphrag_pipeline.py L869-880](../../recon_graphrag/recon_graphrag/pipelines/graphrag_pipeline.py#L869)). LLM rescue tidak bisa dimatikan tanpa keluar dari pipeline. |
| R2-5 | (tidak dibahas) | Conflict pada satu pasangan memblokir **seluruh grup**: grup normalized berisi 3 node dengan satu yang berkonflik tidak menghasilkan merge apa pun (§6.7). |
| R2-6 | Ketergantungan urutan hanya pada batas kandidat 20 (§4.7). | Grouping star juga bergantung urutan load. Data yang sama menghasilkan 1 atau 2 node tergantung entity mana yang jadi anchor (§6.5). |
| R2-7 | (tidak dibahas) | Pada hybrid dalam satu build, ringkasan LLM saat merge (`_prepare_merge_summaries`) **ditimpa** oleh DescriptionSummarizer. Hasilnya 2 panggilan LLM untuk satu node, dan yang pertama terbuang (§4.8). |
| R2-8 | (tidak dibahas) | *(pembacaan kode)* Pasca-merge di Neo4j, deskripsi **semua** relasi node survivor ditulis ulang menjadi gabungan mentah, termasuk relasi yang sudah diringkas (status tetap `success`) dan relasi `FROM_CHUNK`/`SUBJECT_OF` (§4.10). |
| R2-9 | (tidak dibahas) | *(pembacaan kode)* ID sama dengan label berbeda lintas dokumen menghasilkan **dua node ber-`id` identik**. `MERGE` writer mencocokkan label, sedangkan `MATCH` evidence dan relasi hanya mencocokkan `id`, sehingga keduanya mendapat relasi. Resolver yang label-scoped tidak pernah menggabungkannya (§2.1). |
| R2-10 | Angka §6.2 (27 entity) dan §6.4 (skala). | Dataset dan generator versi sebelumnya tidak tersimpan, sehingga angka absolutnya tidak bisa direproduksi. §6.2 dan §6.4 diganti dengan hasil re-run (dataset tercantum). Kesimpulan kualitatifnya tetap sama. §6.1, §6.3, §6.6 (lama), dan set 12 lokasi tereproduksi **persis**. |
| R2-11 | "49 unit test lulus". | Benar, dengan syarat `pytest-asyncio` terpasang (lihat metode verifikasi). |

### Revisi ke-1

| # | Klaim versi awal | Fakta setelah verifikasi |
|---|---|---|
| 1 | "Tidak ada resolution saat ingest." | Ada dedup exact saat ingest lewat ID `"<label>:<normalized-name>"` → `uuid5` → `MERGE` (§2.1). |
| 2 | Blocking: "token pertama = seluruh string", "akronim tidak pernah terdeteksi". | Blocking efektif = filter panjang + huruf pertama sama. Akronim gugur di filter panjang, token-swap gugur di huruf pertama (§3.3). |
| 3 | Alias: "canonical harus ada", "hanya pasangan". | Benar. Selain itu alias dilewati untuk node yang sudah tergabung dan bisa tidak pernah konvergen (§4.2). |
| 4 | Normalisasi hanya "berisiko". | Strategi default `normalized` sendiri menghasilkan false merge (§4.3). |
| 5 | "1 LLM call per pasangan". | Belasan panggilan LLM per entity per build karena LLM rescue (§4.5). |
| 6–9 | (tidak dibahas) | Batas kandidat 20; status summary pasca-merge; parameter tidak bisa diatur lewat pipeline; perbedaan Neo4j/Memgraph. |

---

## 1. Posisi dalam pipeline

```
Chunk ─► LLM extraction (prompt: id "<label>:<normalized-name>")      ← L0: normalisasi oleh LLM
      ─► parser / SchemaValidator (name := identity_property|name|title|id)
      ─► assembler: canonical_key = node.id → entity_id = uuid5(graph_name, canonical_key)
      ─► writer: MERGE (e:__Entity__:Label {id, graph_name})          ← L1: dedup exact saat ingest
      ─► [_finalize_graph, SELURUH graph_name]
            backfill descriptions
            → entity resolution (strategi: exact | normalized* | fuzzy | hybrid)   ← L2: post-hoc
            → DescriptionSummarizer (LLM) → embedding → validasi
(* default)
```

## 2. Identitas dan orkestrasi

### 2.1 L0/L1: identitas saat ingest

- Prompt rule 6–7: *"Every node must have a stable id"* dan *"Prefer IDs in this format: `<label>:<normalized-name>`"* ([prompts.py L150-151](../../recon_graphrag/recon_graphrag/extraction/prompts.py#L150)). Formatnya hanya anjuran ("Prefer"). Prompt gleaning mengulang format yang sama dan memberi daftar node yang sudah diekstrak.
- Parser hanya mengambil `id` mentah dari LLM, **tanpa normalisasi kode** ([parser.py L19-27](../../recon_graphrag/recon_graphrag/extraction/parser.py#L19)). Saat gleaning, node baru disaring dengan `n.id not in existing_ids`, yaitu exact string ([extractor.py L56-63](../../recon_graphrag/recon_graphrag/extraction/extractor.py#L56)).
- Validator membuang properti di luar schema kecuali `name/description/weight` ([validator.py L82-90](../../recon_graphrag/recon_graphrag/extraction/validator.py#L82)). `name` diisi dari `identity_property` (default `"name"`) → `name` → `title` → `id` ([L32-38](../../recon_graphrag/recon_graphrag/extraction/validator.py#L32)).
- `canonical_key = node.properties.canonical_key or node.id` ([assembler.py L59](../../recon_graphrag/recon_graphrag/extraction/assembler.py#L59)). Karena validator membuang `canonical_key` (kecuali dideklarasikan di schema), nilainya pada ekstraksi teks praktis selalu `node.id` mentah dari LLM, peka kapitalisasi. Dalam satu dokumen, entity dikelompokkan per `canonical_key`. Properti **dan label** dari mention pertama yang menang.
- Jalur structured ingestion memakai key `"{label}:{nilai}"` ([structured.py L53-54](../../recon_graphrag/recon_graphrag/extraction/structured.py#L53)) tanpa normalisasi. Jadi `Organization:ICCN` (structured) dan `organization:iccn` (LLM) menghasilkan UUID berbeda, dan baru bisa bertemu di resolusi post-hoc lewat `name`.
- `entity_id = uuid5("recon-graphrag/entity/{graph_name}/{canonical_key}")` ([L202](../../recon_graphrag/recon_graphrag/extraction/assembler.py#L202)), lalu writer menjalankan `MERGE (e:__Entity__:{label} {id, graph_name}) SET e += row.properties` ([writer.py L88](../../recon_graphrag/recon_graphrag/pipelines/writer.py#L88)). Daftar `descriptions` digabung dan dideduplikasi, sementara `name`/`title` mengikuti tulisan terakhir.
- **Implikasinya:** dedup exact lintas chunk dan lintas dokumen sudah terjadi saat tulis, **asalkan LLM menghasilkan ID yang sama persis**. Prompt tidak memberi aturan tentang akronim, urutan kata, atau bentuk panjang/pendek.
- **Kasus tepi label** *(pembacaan kode)*: `MERGE` writer mencocokkan pola lengkap termasuk label. Bila dokumen lain memberi ID yang sama dengan label berbeda, terbentuk **node kedua dengan `id` identik**. Penulisan evidence ([writer.py L144-146](../../recon_graphrag/recon_graphrag/pipelines/writer.py#L144)) dan relasi ([L161-163](../../recon_graphrag/recon_graphrag/pipelines/writer.py#L161)) memakai `MATCH (e:__Entity__ {id, graph_name})` tanpa label, sehingga keduanya ikut tertaut. Resolver mengelompokkan per label, jadi pasangan ini tidak pernah dibandingkan. Tidak ada uniqueness constraint pada `__Entity__.id` (index manager hanya membuat constraint untuk community).

### 2.2 Interaksi L1 dengan merge post-hoc

Saat resolver menggabungkan grup, node survivor mendapat `id = entity_id` milik canonical ([neo4j/entity_resolution.py L111-114](../../recon_graphrag/recon_graphrag/graphdb/neo4j/entity_resolution.py#L111), [memgraph L91](../../recon_graphrag/recon_graphrag/graphdb/memgraph/entity_resolution.py#L91)). UUID dan `canonical_key` node yang diserap hilang. Yang tersimpan hanya *nama*-nya di `aliases`, dan properti `aliases` tidak dibaca oleh writer, extractor, maupun resolver. Akibatnya, dokumen baru yang kembali menyebut ID yang sudah diserap akan **membuat node duplikat baru**. Node itu baru tergabung lagi (jika strateginya mampu) pada finalize berikutnya. Antara ingest dan finalize, retrieval melihat duplikat.

### 2.3 Orkestrasi dan parameter

- `GraphBuilderPipeline` default: `perform_entity_resolution=True`, `entity_resolution_strategy="normalized"`, `allow_ai_auto_merge=False`, `fail_on_resolution_error=False` ([graphrag_pipeline.py L67-93](../../recon_graphrag/recon_graphrag/pipelines/graphrag_pipeline.py#L67)).
- `_finalize_graph` ([L655](../../recon_graphrag/recon_graphrag/pipelines/graphrag_pipeline.py#L655)) berjalan atas **seluruh** `graph_name`. Tidak ada mode inkremental. `build_from_documents` menjalankannya sekali per batch, sedangkan `build_from_text`/`build_from_rows` menjalankannya **setiap panggilan** (`finalize=True`, [L180](../../recon_graphrag/recon_graphrag/pipelines/graphrag_pipeline.py#L180)). Pemanggil yang meng-ingest dokumen satu per satu lewat `build_from_text` membayar resolusi seluruh graph per dokumen.
- `_resolve_entities` ([L853](../../recon_graphrag/recon_graphrag/pipelines/graphrag_pipeline.py#L853)) **tidak meneruskan** `merge_threshold`, `review_threshold`, maupun `max_candidates_per_entity`. Lewat pipeline nilainya terkunci di 95/85/20 (dan ambang confidence auto-merge = `merge_threshold/100` = 0,95). Nilai ini hanya bisa diubah dengan memanggil `store.resolve_entities_*` langsung ([neo4j/store.py L129](../../recon_graphrag/recon_graphrag/graphdb/neo4j/store.py#L129)). Konkurensi LLM resolver juga terkunci di 5.
- Untuk `hybrid`, pipeline **selalu** meneruskan `embedder=self.embedder` dan `llm=self.llm` ([L869-880](../../recon_graphrag/recon_graphrag/pipelines/graphrag_pipeline.py#L869)). Akibatnya LLM review + rescue, embedding nama, dan konsolidasi deskripsi via LLM selalu aktif. "Hybrid tanpa LLM" (misalnya alias + conflict saja) tidak bisa dijalankan lewat pipeline.
- `review_groups` hanya di-log di level DEBUG dan dihitung di ringkasan stdout ("N flagged for review"). Build mengembalikan `{"extraction", "validation"}` tanpa hasil resolusi. **Antrean review tidak sampai ke pemanggil** dan tidak dipersistensi.
- Error resolusi ditelan (`fail_on_resolution_error=False`). Merge dieksekusi per grup tanpa transaksi bersama, sehingga kegagalan di tengah meninggalkan sebagian grup ter-merge.

### 2.4 Eksekusi merge per backend

| Aspek | Neo4j ([L71](../../recon_graphrag/recon_graphrag/graphdb/neo4j/entity_resolution.py#L71)) | Memgraph ([L54](../../recon_graphrag/recon_graphrag/graphdb/memgraph/entity_resolution.py#L54)) |
|---|---|---|
| Mekanisme | `apoc.refactor.mergeNodes(properties:'combine', mergeRels:true)` | `_combine_properties` (meniru `combine`) + `_rewire_relationships` manual |
| Prasyarat | APOC. Tanpa APOC, resolusi di-skip (`_preflight`), kecuali dry-run. | – |
| Survivor | Nama **terpanjang** (sort stabil, seri → urutan load) | sama |
| Properti konflik | Digabung menjadi list (`combine`) | sama (terverifikasi: `title`, `created` jadi list) |
| Relasi antar anggota grup | Tidak ada langkah penghapusan. *(pembacaan kode)* Mengikuti default APOC, relasi ini menjadi self-loop. | **Dihapus** eksplisit (ada test `self_loop_guards`) |
| Deskripsi relasi pasca-merge | *(pembacaan kode)* **Semua** relasi survivor ditulis ulang: `descriptions = toSet(descriptions + [description])`, `description` = gabungan mentah, `description_summary_fallback=true` ([L149-181](../../recon_graphrag/recon_graphrag/graphdb/neo4j/entity_resolution.py#L149)) | Hanya relasi yang di-rewire. Relasi baru hasil `MERGE` kehilangan list observasi dan menyisakan `[description]` ([L203-227](../../recon_graphrag/recon_graphrag/graphdb/memgraph/entity_resolution.py#L203)) |
| `aliases` | `coalesce(n.aliases, []) + $aliases`, tanpa dedup ([L189](../../recon_graphrag/recon_graphrag/graphdb/neo4j/entity_resolution.py#L189)) | dedup `dict.fromkeys` ([L109-111](../../recon_graphrag/recon_graphrag/graphdb/memgraph/entity_resolution.py#L109)) |
| Pasca-merge lain | Re-wrap `source_chunk_ids` skalar, `embedding=null`, tulis `descriptions`/`description`/fingerprint | `embedding=None`, tulis ringkasan merge |

---

## 3. Strategi resolusi post-hoc

Semua strategi memuat **seluruh** `__Entity__` dengan `graph_name` dan `name` tidak null ke memori, lalu mengelompokkan per `(graph_name, domain_label)`. `domain_label` = label pertama selain `__Entity__`.

### 3.1 Normalisasi — [`_normalize_name` L51](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L51)

NFC (bukan NFKC) → casefold → hapus kata suffix badan usaha Inggris (`corp, corporation, inc, incorporated, ltd, limited, llc, co, company, plc, llp, lp, holdings, group`) di **posisi mana pun** → hapus semua tanda baca → **hapus semua spasi**. Hasilnya bisa string kosong, dan tidak ada guard untuk itu.

### 3.2 `exact` / `normalized` — [L397](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L397), [L407](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L407)

Grup per nilai mentah atau nilai ternormalisasi, lalu langsung di-merge. **Tanpa review, tanpa conflict check** (conflict properties hanya berlaku di `hybrid`).

### 3.3 `fuzzy` — [`_build_fuzzy_groups` L417](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L417)

1. Bentuk grup normalized. Anggotanya **dikeluarkan** dari fuzzy ([L425-426](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L425)).
2. Untuk tiap anchor singleton (urutan load), ambil kandidat lewat [`_blocked_candidates` L479](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L479) dengan pemindaian O(n) per anchor (O(n²) total):
   - label dan graph sama;
   - `|len1−len2| ≤ max(3, 0.4·min_len)` atas string tanpa spasi;
   - "token pertama sama ATAU `_is_acronym_match`". Karena string tidak berspasi, token pertama = seluruh string, dan `_is_acronym_match` ([L509](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L509)) hanya membandingkan **huruf pertama**. Efektifnya: **huruf pertama sama**;
   - `candidates[:20]` menurut **urutan pool** ([L506](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L506)), bukan skor.
3. Skor `fuzz.ratio` (Levenshtein ternormalisasi, sensitif urutan) atas string tanpa spasi:
   - ≥95 → masuk grup anchor (*star*, tidak transitif). Kandidat yang sudah masuk grup ditandai `used` dan tidak pernah jadi anchor;
   - ≥85 → `review_groups` (`fuzzy_candidate`);
   - selain itu → `dropped_pairs`.

### 3.4 `hybrid` — [`_build_hybrid_groups` L520](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L520)

Urutan eksekusi:
1. fuzzy (termasuk normalized);
2. **alias** ([L698](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L698)) hanya untuk node yang **belum** ada di grup mana pun, dicocokkan lewat `name` ternormalisasi (bukan properti `aliases`);
3. **conflict rules** ([L606](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L606)) untuk grup (normalized/fuzzy/alias) dan review fuzzy. Konflik pada satu pasangan memblokir seluruh grup ([L618-623](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L618)). Nilai list hanya dibandingkan elemen pertamanya;
4. **embedding** ([L772](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L772)) hanya untuk review aktif, dengan 2 panggilan per pasangan berbasis nama saja dan tanpa cache. Skornya **tidak dipakai keputusan apa pun** dan hanya ikut di payload prompt;
5. **LLM rescue** ([L572](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L572), [L805](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L805)): **semua** `dropped_pairs` menjadi review, tanpa skor embedding dan **tanpa conflict check** (sudah lewat tahap 3);
6. **LLM review** ([L847](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L847)): 1 panggilan per pasangan. Payload berisi seluruh `review_group` (termasuk skor dan node id), profil entity, `user_aliases` lengkap, dan `llm_guidance`. Profil ([entity_resolution_context.py L52-83](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution_context.py#L52)) memuat `name/title/description/canonical_key/human_readable_id/aliases` (string dipotong 500 karakter). Pada `safe_defaults`, profil juga memuat **semua** properti lain yang tidak ada di `INTERNAL_CONTEXT_FIELDS`, termasuk field bookkeeping summarizer (§6.8);
7. **auto-merge** ([L648](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L648)) hanya jika `allow_ai_auto_merge=True`, `same_entity is True`, `merge_allowed is True`, `confidence ≥ merge_threshold/100`, dan tidak ada node yang sudah ter-merge. **Tidak ada pemeriksaan konflik di sini.**

Deskripsi grup yang di-merge dikonsolidasi oleh [`_prepare_merge_summaries` L308](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L308): lewat LLM bila `llm` diberikan (hanya hybrid), selain itu gabungan `\n`.

---

## 4. Kelemahan dan bug (diverifikasi, urut keparahan)

### 4.1 [KRITIS] Blocking menolak semua kasus target sebelum fuzzy/LLM sempat menilai
Blocking efektif = filter panjang + huruf pertama sama (§3.3). Akibatnya (§6.1):
- akronim vs bentuk panjang gugur di filter panjang (ICCN, Kemenparekraf, Kemenkeu, Bekraf, BI);
- token-swap gugur di huruf pertama (Kota Malang/Malang Kota), begitu juga prefiks ("PT Bank …" vs "Bank …").

`dropped_pairs` hanya berisi pasangan yang lolos blocking, jadi **LLM rescue pun tidak pernah melihatnya**. Dengan LLM *oracle* yang selalu benar, recall sama persis dengan fuzzy tanpa LLM (2/10, §6.2). Sebaliknya, pasangan berbeda yang kebetulan berhuruf awal sama dan panjangnya mirip (Kota Malang/Kota Madiun, Kementerian Keuangan/Kelautan, Jawa Timur/Barat) justru lolos ke LLM.

### 4.2 [KRITIS] Alias dictionary rapuh dan bisa tidak pernah konvergen
Terverifikasi (§6.3):
- (a) Canonical yang tidak ada sebagai node → tidak ada merge.
- (b) Satu canonical dengan dua alias → hanya **satu** alias per run (pasangan + `break`, [L736-741](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L736)).
- (c, d) Alias **dilewati** untuk node yang sudah tergabung di normalized/fuzzy ([L718](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L718), [L726](../../recon_graphrag/recon_graphrag/graphdb/entity_resolution.py#L726)).
- **Tidak konvergen** bila survivor (nama terpanjang) ≠ nama canonical. Contohnya bentuk jamak "Networks", bentuk berkurung "… (ICCN)", atau user memakai akronim sebagai canonical. Setelah build pertama nama canonical lenyap, dan karena pencocokan hanya memakai `name`, alias sisanya **tidak pernah** tergabung di build mana pun.
- Alias juga tidak dipakai saat ingest (§2.2), dan kompleksitasnya O(N²·|alias|) dengan `_normalize_name` dihitung ulang di loop terdalam.

### 4.3 [TINGGI] Normalisasi default menghasilkan false merge tanpa review
Strategi default `normalized` langsung merge tanpa conflict check (§6.2):
- "Lampiran 1.2" = "Lampiran 12", "Tabel 2.1" = "Tabel 21": tanda baca dan spasi dihapus, sehingga penomoran desimal dan bertingkat menyatu. Ini risiko nyata di korpus regulasi.
- "LP Cipinang" = "Cipinang": `lp` dianggap suffix *limited partnership*, padahal di Indonesia LP = Lembaga Pemasyarakatan.
- "Astra Group" = "Astra" dan "Lippo Group" = "Lippo": ambigu. `group` dibuang di posisi mana pun, termasuk di awal ("Co-operative Bank" → "operativebank").
- **Nama yang ternormalisasi menjadi string kosong** ("CO", "LLC", "Group", "Co.", "LP") tergabung semua dalam satu label. Terverifikasi: "CO", "LLC", dan "Group" menjadi satu node "Group".
- Penghapusan spasi menyatukan token berbeda ("Kali Mantan" = "Kalimantan").

Suffix badan usaha Indonesia (`PT`, `Tbk`, `CV`, `(Persero)`) justru tidak dikenali, dan "Grup" (ejaan Indonesia) tidak sama dengan "Group".

### 4.4 [TINGGI] Veto `conflict_properties` bisa dilewati, dan tidak ada guard numerik
- **Bypass veto lewat rescue** (R2-1): pasangan `llm_rescue` ditambahkan setelah `_apply_conflict_rules`, dan `_promote_ai_reviews` tidak memeriksa konflik. Dengan `allow_ai_auto_merge=True`, satu keputusan LLM yang keliru cukup untuk menggabungkan dua entity yang propertinya jelas berbeda. Contoh konfigurasi di dokumentasi recon sendiri (`Person: ["birth_date"]` + auto-merge) terkena masalah ini. Jalur review fuzzy (85–95) tetap diblokir dengan benar (§6.7).
- **Tanpa guard numerik**: di fuzzy, pasangan yang hanya berbeda angka mendapat rasio tinggi. "UU No. 24/25 Tahun 2019" bernilai 93,3 dan "… Pasal 9/10" bernilai 93,0, keduanya masuk review. Pada nama yang lebih panjang, rasio ≥95 langsung **auto-merge tanpa review** (data sintetis: "Badan Pendidikan Kabupaten Kediri 4" = "… 36", 22 grup salah pada n=1000, §6.4).
- Conflict properties hanya berlaku di hybrid dan hanya bila nomor disimpan sebagai properti terstruktur. Konflik juga memblokir seluruh grup, bukan hanya anggota yang berkonflik (§6.7).

### 4.5 [TINGGI] Biaya LLM eksplosif dan diulang setiap build
LLM rescue mengirim **semua** pasangan yang lolos blocking, dengan batas atas ≈ `max_candidates_per_entity × n` = 20·n per build. Terukur 9–17 panggilan LLM per entity (n=1000 → 16.218 panggilan, §6.4). Lewat pipeline, rescue selalu aktif untuk hybrid (§2.3). Keputusan tidak di-cache atau dipersistensi, sehingga semuanya diulang di setiap build (dan di setiap `build_from_text`). Seluruh dict `user_aliases` dan field bookkeeping (§6.8) juga ikut dikirim di setiap prompt.

### 4.6 [SEDANG] Skalabilitas O(n²) Python
Waktu fuzzy grouping 0,75 detik (n=500) → 2,6 detik (n=1000) → 10,6 detik (n=2000) → 40 detik (n=4000) untuk satu label. Pertumbuhannya kuadratik. Seluruh entity dimuat ke memori setiap build.

### 4.7 [SEDANG] Hasil bergantung urutan load
- `candidates[:20]` diambil menurut urutan pool. Typo yang jelas ("Dinas Pariwisata Kota Malang" vs "…Malangg", rasio ±98) **terlewat** bila ≥20 entity berbeda (huruf awal sama, panjang mirip) dimuat sebelum keduanya. Data yang sama dengan urutan berbeda tetap ter-merge (§6.5).
- Grouping star memilih anchor menurut urutan load. Data A~B~C yang sama menghasilkan 1 atau 2 node tergantung siapa yang jadi anchor (§6.5).
- Urutan `MATCH` di Neo4j/Memgraph tidak dijamin, jadi hasilnya tidak deterministik antar database atau antar restore.

### 4.8 [SEDANG] Kualitas dan biaya deskripsi setelah merge
- **Lintas build, non-hybrid:** node lama yang sudah diringkas (`description_summary_status='success'`) yang digabung dengan duplikat baru tetap berstatus `success`. Penyebabnya: node baru tidak membawa properti status ([writer.py L112-116](../../recon_graphrag/recon_graphrag/pipelines/writer.py#L112)), sehingga `combine` mempertahankan nilai lama. DescriptionSummarizer hanya memilih node dengan status ≠ `success` ([neo4j/store.py L433](../../recon_graphrag/recon_graphrag/graphdb/neo4j/store.py#L433)), jadi deskripsi node itu **tetap gabungan mentah** yang juga mencampur ringkasan lama sebagai "observasi" (§6.6).
- **Dalam satu build, hybrid:** `_prepare_merge_summaries` memanggil LLM untuk meringkas grup. Karena status node masih null, DescriptionSummarizer meringkas ulang dari `descriptions` dan menimpa hasil pertama. Akibatnya 2 panggilan LLM per node hasil merge, dan yang pertama terbuang.

### 4.9 [SEDANG] Fitur review tidak dapat dipakai lewat pipeline
`review_groups` tidak dikembalikan oleh build dan tidak disimpan (§2.3). Mode default *audit-only* (`allow_ai_auto_merge=False`) berarti ribuan panggilan LLM menghasilkan keputusan yang hanya berakhir di log DEBUG.

### 4.10 [RENDAH] Lain-lain
- Profil LLM pada `safe_defaults` membocorkan field bookkeeping (hash fingerprint, timestamp, status, `descriptions` duplikat), bertentangan dengan dokumentasi recon (§6.8). Akibatnya token terbuang dan muncul noise yang bisa memengaruhi keputusan.
- *(pembacaan kode)* Pasca-merge di Neo4j, deskripsi semua relasi survivor ditulis ulang menjadi gabungan mentah dengan `description_summary_fallback=true`. Untuk relasi yang sudah diringkas, statusnya tetap `success` sehingga tidak diringkas ulang. Relasi `FROM_CHUNK`/`SUBJECT_OF` ikut mendapat `descriptions=[]`/`description=''`. Di Memgraph, relasi hasil rewire kehilangan list observasinya (§2.4).
- *(pembacaan kode)* ID sama + label berbeda menghasilkan node kembar ber-`id` identik yang tidak pernah di-resolve (§2.1).
- Embedding hanya berbasis nama, tidak di-batch, tidak di-cache, dan tidak dipakai untuk blocking maupun keputusan.
- Survivor "nama terpanjang" bisa memilih varian yang bertele-tele dan menimpa nama canonical pilihan user.
- `same_entity` harus boolean `true`; string `"true"` tidak dianggap setuju. Parsing JSON memakai regex, tanpa structured output, dan error LLM tidak di-retry.
- Perilaku relasi antar anggota grup dan dedup `aliases` berbeda antara Neo4j dan Memgraph (§2.4).

---

## 5. Kekuatan (layak diadopsi)

1. **Dedup exact saat ingest lewat UUID deterministik** dari ID, plus `MERGE` yang idempoten. Polanya tepat. Yang kurang: normalisasi deterministik di kode (tidak diserahkan ke LLM) dan uniqueness constraint.
2. **Arsitektur strategi berjenjang** (exact → normalized → fuzzy → hybrid) dengan sinyal yang dilaporkan (`signals`).
3. **Konsep alias dictionary global + per-label**: bentuk konfigurasi yang tepat untuk kustomisasi, walaupun implementasinya rapuh (§4.2).
4. **Konsep `conflict_properties` sebagai veto** yang juga berlaku pada grup deterministik (dalam hybrid). Terbukti memblokir pasangan review fuzzy, tetapi implementasinya bocor untuk pasangan rescue (§4.4). Bila diadopsi, veto harus dievaluasi tepat sebelum merge, untuk semua jalur.
5. **Profil konteks untuk LLM** (nama, label, deskripsi, alias, properti terpilih) dengan `context_mode`/`context_properties`, serta `llm_guidance` domain. Desain prompt review yang baik. Default `safe_defaults` perlu diganti dengan allowlist (§6.8).
6. **Grouping star (non-transitif)** di fuzzy: konservatif terhadap chain collapse. Terbukti mencegah rantai di §6.2 (set 12 lokasi).
7. **Default aman untuk LLM**: audit-only, auto-merge butuh tiga syarat dan confidence ≥ 0,95.
8. **Detail merge yang matang**: re-wrap list APOC, re-embedding, merge relasi duplikat dengan union `source_chunk_ids`, fingerprint input deskripsi, penghapusan self-loop (Memgraph).
9. **Teruji**: 49 unit test resolver lulus (dengan `pytest-asyncio`), walaupun jalur yang bermasalah di atas tidak tercakup.

---

## 6. Eksperimen (kode asli recon)

Harness memakai `BaseEntityResolver` asli dengan store in-memory (§ metode verifikasi). LLM oracle = hanya setuju untuk pasangan ground truth. LLM "selalu ya" = menyetujui semua pasangan dengan confidence 0,99.

### 6.1 Semantik blocking (`_blocked_candidates` asli)

| A | B | Lolos filter panjang | Huruf awal sama | Lolos blocking | `fuzz.ratio` |
|---|---|---|---|---|---|
| ICCN | Indonesia Creative Cities Networks (ICCN) | Tidak | Ya | **Tidak** | 20,5 |
| Kemenparekraf | Kementerian Pariwisata Dan Ekonomi Kreatif | Tidak | Ya | **Tidak** | 51,0 |
| Bekraf | Badan Ekonomi Kreatif | Tidak | Ya | **Tidak** | 48,0 |
| Kemenkeu | Kementerian Keuangan | Tidak | Ya | **Tidak** | 59,3 |
| BI | Bank Indonesia | Tidak | Ya | **Tidak** | 26,7 |
| Kota Malang | Malang Kota | Ya | **Tidak** | **Tidak** | 60,0 |
| Kota Malang | Kabupaten Malang | Tidak | Ya | Tidak | 64,0 |
| Kota Malang | Kota Madiun ❌ | Ya | Ya | Ya → rescue LLM | 70,0 |
| Kementerian Keuangan | Kementerian Kelautan ❌ | Ya | Ya | Ya → rescue LLM | 84,2 |
| Jawa Timur | Jawa Barat ❌ | Ya | Ya | Ya → rescue LLM | 55,6 |
| UU No. 24 Tahun 2019 | UU No. 25 Tahun 2019 ❌ | Ya | Ya | Ya → review | 93,3 |
| UU … Pasal 9 | UU … Pasal 10 ❌ | Ya | Ya | Ya → review | 93,0 |
| Indonesia Creative Cities Network | … Networks | Ya | Ya | Ya → **merge** | 98,4 |

(❌ = pasangan berbeda.)

### 6.2 Normalisasi dan end-to-end

**Normalisasi (`_normalize_name` asli):**

| Input A | Input B | Hasil |
|---|---|---|
| Lampiran 1.2 | Lampiran 12 | sama (`lampiran12`) ❌ |
| Tabel 2.1 | Tabel 21 | sama ❌ |
| LP Cipinang | Cipinang | sama (`cipinang`) ❌ |
| Astra Group / Lippo Group | Astra / Lippo | sama |
| Co-operative Bank | Operative Bank | sama ❌ |
| CO / Group / Co. / LLC | (satu sama lain) | semua `''` → tergabung ❌ |
| Kali Mantan | Kalimantan | sama |
| PT Astra International Tbk | Astra International | **beda** (suffix ID tidak dikenali) |
| Grup Lippo | Lippo Group | **beda** |

**End-to-end: 34 entity, 3 label, 10 pasangan benar.** Pasangan benar: 3 akronim (ICCN, Kemenparekraf, Bekraf), PT BRI (Persero) Tbk/BRI, Kota Malang/Malang Kota, Jawa Timur/Jatim, DI Yogyakarta, UU No. 24/Undang-Undang Nomor 24, Charter/Charters, typo Malang/Malangg. Pengecoh: Lampiran 1.2/12, LP Cipinang, Astra Group, UU 24/25, Pasal 9/10, Kemenkeu/Kelautan, Jatim/Jabar, Bank Indonesia/Tbk.

| Konfigurasi | Pasangan benar ter-merge | Pasangan false merge | Panggilan LLM |
|---|---|---|---|
| `normalized` (default) | 0/10 | **3** (Lampiran 1.2=12, LP Cipinang=Cipinang, Astra Group=Astra) | 0 |
| `fuzzy` | 2/10 | 3 | 0 |
| `hybrid` + alias (ICCN, Kemenparekraf, Kota Malang) | 5/10 | 3 | 0 |
| `hybrid` + LLM oracle, auto-merge on | **2/10** (sama dengan fuzzy) | 3 | 12 |
| `hybrid` + LLM "selalu ya", auto-merge on | 2/10 | **9** | 12 |
| `hybrid` + alias + LLM oracle + auto-merge | 5/10 | 3 | 12 |

LLM oracle tidak menambah satu pun pasangan benar, karena semua kasus target gugur di blocking (§4.1). Alias hanya membantu pasangan yang dikonfigurasi persis dan tidak terkena masalah §4.2. (Versi sebelumnya memakai set 27 entity yang datasetnya tidak tersimpan, dengan hasil 1/2/3/2/2/3 dari 10. Kesimpulannya sama.)

**Set 12 entity lokasi** (sama dengan uji neo4j-graphrag, [03 §10.5](03-neo4j-graphrag-python.md)): Kota Malang, Malang Kota, Kabupaten Malang, Malang, Pemerintah Kota Malang, Kota Surabaya, Kota Batu, Kota Bandung, Jawa Timur, Jawa Barat, Provinsi Jawa Timur, Kota Yogyakarta. Hanya Kota Malang/Malang Kota yang sama, jadi idealnya 11 node. Hasil tereproduksi persis:

| Konfigurasi | Node akhir | Catatan |
|---|---|---|
| `normalized` (default) | 12 | Tidak ada merge |
| `fuzzy` | 12 | Tidak ada merge |
| `hybrid` + LLM oracle + auto-merge | 12 | Kota Malang↔Malang Kota tidak pernah sampai ke LLM (huruf pertama beda). 13 panggilan LLM. |
| `hybrid` + LLM "selalu ya" + auto-merge | 8 | 4 false merge **berpasangan**: Kota Malang=Kota Surabaya, Kabupaten Malang=Kota Bandung, Pemerintah Kota Malang=Provinsi Jawa Timur, Jawa Barat=Jawa Timur. Grouping star mencegah rantai. |

### 6.3 Alias lintas build (hybrid tanpa LLM, diiterasi sampai stabil)

| Skenario | Jumlah node per build | Hasil akhir |
|---|---|---|
| Canonical tidak ada sebagai node | 2 → 2 | Tidak ada merge |
| Canonical + 2 alias | 3 → 2 → 1 | Satu alias per build, konvergen di build 2 |
| Canonical punya kembaran beda kapitalisasi | 3 → 2 → 1 | Alias dilewati di build 1, konvergen di build 2 |
| Canonical punya kembaran jamak ("Networks") | 3 → 2 → 2 | **Tidak konvergen**: "…Networks" + "ICCN" tetap 2 node |
| Alias lebih panjang dari canonical cocok duluan | 3 → 2 → 2 | **Tidak konvergen**: "… (ICCN)" + "ICCN" tetap 2 node |
| User memakai "ICCN" sebagai canonical (2 alias) | 3 → 2 → 2 | **Tidak konvergen**: tersisa 2 node |
| Alias per-label (`{"Org": {...}}`) | 2 → 1 | Konvergen |

### 6.4 Skala (generator nama instansi sintetis, satu label, semua entity berbeda)

Pola nama: `{Badan|Dinas|Kantor|Balai|Bidang} {8 sektor} {Kota|Kabupaten} {12 kota} {1–60}`. LLM selalu menjawab "tidak", sehingga yang dihitung murni jumlah panggilan.

| n | Fuzzy grouping | Grup auto-merge fuzzy (semuanya salah) | Panggilan LLM hybrid | per entity | Panggilan embedding |
|---|---|---|---|---|---|
| 500 | 0,75 s | 7 | 8.674 | 17,3 | 162 |
| 1.000 | 2,64 s | 22 | 16.218 | 16,2 | 282 |
| 2.000 | 10,64 s | 53 | 26.059 | 13,0 | 698 |
| 4.000 | 40,13 s | 88 | 36.689 | 9,2 | 1.180 |

Jumlah panggilan per entity bergantung kepadatan nama mirip dalam dataset, dengan batas atas 20 per entity. Generator versi sebelumnya menghasilkan 17.190 panggilan pada n=1000, orde yang sama. Contoh auto-merge salah: nama yang hanya berbeda nomor di akhir ("… Kediri 4" = "… 36").

### 6.5 Ketergantungan urutan

| Entity mirip berbeda yang dimuat sebelum twin | Typo twin (rasio ±98) ter-merge? | Data sama, twin dimuat duluan |
|---|---|---|
| 19 | Ya | Ya |
| 20 / 21 | **Tidak** | Ya |

Star grouping: tiga varian A, B, C dengan AB=98,0, BC=98,1, AC=96,2 dan `merge_threshold=97`. Urutan [A,B,C] → 2 node, [C,A,B] → 2 node, [B,C,A] → 1 node.

### 6.6 Properti node hasil merge lintas build (semantik `combine` Memgraph, strategi tanpa LLM)

| Properti | Nilai setelah merge |
|---|---|
| `name` | `'Kota Malang'` |
| `title` | `['Kota Malang', 'kota malang']` (list) |
| `created` | `[100, 200]` (list) |
| `description_summary_status` | `'success'` (dari node lama) → summarizer **melewati** node ini |
| `descriptions` | `['obs lama', 'RINGKASAN LAMA', 'obs baru']` |
| `description` | gabungan mentah `\n` |

### 6.7 Conflict properties

| Skenario | Hasil |
|---|---|
| `normalized`, "Lampiran 1.2" vs "12" (nomor beda) | merge (tanpa conflict check) |
| `hybrid` + `conflict_properties={"Regulation": ["nomor"]}`, Lampiran | diblokir (`property_conflict`) |
| `hybrid` + conflict `nomor`, UU No. 24 vs 25 (jalur review fuzzy), LLM "selalu ya", auto-merge | diblokir ✔ |
| `hybrid` + conflict `kode_wilayah`, Kota Malang (3573) vs Kota Madiun (3577) (jalur **rescue**), LLM "selalu ya", auto-merge | **ter-merge** ❌, `kode_wilayah` menjadi `['3573','3577']` |
| Grup normalized "Astra"/"ASTRA" (npwp 1) + "Astra." (npwp 2) | **seluruh grup** diblokir, "Astra"/"ASTRA" yang cocok pun tidak ter-merge |

### 6.8 Isi profil LLM pada `safe_defaults`

Node berbentuk keluaran writer + summarizer menghasilkan `profile["properties"]`:

```json
{"type": "Location", "descriptions": ["obs 1", "obs 2"], "observation_count": 2,
 "description_summary_status": "success", "description_input_fingerprint": "aaaa…(64)",
 "description_summary_updated": 1700000000000, "description_summary_fallback": false}
```

Semua ini bukan fakta domain. `descriptions` mengulang `description`, dan dua field terakhir adalah hash dan timestamp, padahal dokumentasi recon menyatakan timestamp dikecualikan.

---

## 7. Penilaian

| Dimensi | Skor (1–5) | Catatan |
|---|---|---|
| Recall kasus target | 1 | Blocking menolak semua kasus target, sehingga LLM oracle pun tidak menambah recall. Alias bisa tidak konvergen. |
| Precision / keamanan | 2 | Default `normalized` sudah false-merge (desimal, `lp`, string kosong). Tanpa guard numerik. Veto konflik hanya di hybrid dan bocor di jalur rescue. Auto-merge LLM off secara default. |
| Kustomisasi | 3 | Alias global/per-label, conflict/context properties, guidance ada. Threshold tidak bisa diatur dan LLM tidak bisa dimatikan untuk hybrid lewat pipeline. Alias rapuh. |
| Efisiensi | 1 | O(n²) per build atas seluruh graph. Hingga ~17 panggilan LLM per entity per build tanpa cache. Ringkasan merge dobel. Embedding tak terpakai. |
| Auditability | 2 | `review_groups` lengkap, tetapi tidak sampai ke pemanggil lewat pipeline dan tidak dipersistensi. `aliases` disimpan di node. |
| Kecocokan arsitektur dengan hybrid_rag | 4 | Sama-sama Neo4j, open-schema, entity bernama bebas, ID deterministik. |
| Kematangan kode | 3 | Teruji (49 test lulus) dan detail merge matang, tetapi bug fundamental di blocking, alias, dan urutan conflict/rescue tidak tertangkap test. Proyek berlabel WIP. |

**Kesimpulan:** recon_graphrag punya **kerangka konsep** yang paling dekat dengan kebutuhan hybrid_rag. Yang patut diadopsi: ID deterministik + `MERGE` saat ingest, strategi berjenjang, alias global/per-tipe, conflict properties sebagai veto, profil LLM berbasis deskripsi dengan guidance domain, grouping star, dan default audit-only.

Implementasinya tidak bisa dipakai apa adanya, karena:
- normalisasi menghapus spasi, memakai suffix Inggris, dan tidak menjaga string kosong, sehingga terjadi false merge;
- blocking efektif hanya "huruf pertama + panjang";
- alias dilewati dan bisa tidak konvergen;
- veto konflik tidak berlaku untuk pasangan rescue;
- tidak ada guard numerik;
- LLM rescue O(n) per entity tanpa cache;
- hasil bergantung urutan load;
- keputusan review tidak tersimpan.

Untuk hybrid_rag, yang dipindahkan adalah konsepnya dengan perbaikan:
- normalisasi deterministik di kode yang mempertahankan token dan angka, menolak key kosong, dan mengenali suffix Indonesia;
- blocking berbasis indeks (alias key, fulltext, ANN), bukan huruf pertama;
- alias registry yang juga dipakai **saat ingest**, dengan canonical milik user yang tidak bisa ditimpa aturan "nama terpanjang";
- veto konflik dan digit guard yang dievaluasi tepat sebelum setiap merge (deterministik maupun LLM);
- urutan deterministik (sort by key);
- LLM batch dengan cache keputusan;
- antrean review yang dipersistensi;
- allowlist field untuk profil LLM.
