# Analisis Entity Merging — `neo4j-graphrag-python`

> Versi yang dianalisis: **neo4j-graphrag 1.22.0** (terinstall di `.venv`, sudah menjadi dependency `ingestion` hybrid_rag).
> Lingkup kode:
> [components/resolver.py](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py) (seluruh mekanisme resolusi),
> [components/entity_relation_extractor.py](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/entity_relation_extractor.py) (penetapan ID node),
> [components/kg_writer.py](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/kg_writer.py) dan [neo4j_queries.py](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/neo4j_queries.py) (penulisan ke Neo4j),
> [components/lexical_graph.py](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/lexical_graph.py) (provenance chunk),
> [simple_kg_builder.py](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/experimental/pipeline/config/template_pipeline/simple_kg_builder.py) (wiring pipeline).
> Clone `neo4j-graphrag-python/` di root repo sedikit lebih baru (bagian "Next" di CHANGELOG), tetapi `resolver.py`, `kg_writer.py`, `neo4j_queries.py`, `entity_relation_extractor.py`, `lexical_graph.py`, dan `types.py` **identik byte-per-byte** dengan versi terinstall. Perbedaan di `simple_kg_builder.py`/`prompts.py` hanya kosmetik.
>
> **Metode verifikasi (revisi 2026-10-17).** Semua klaim perilaku diuji dengan kode package asli pada **Neo4j 5.26 + APOC 5.26.32** lokal (Docker), bukan sekadar simulasi Python:
> - E1: `SimpleKGPipeline` asli end-to-end (LLM dan embedder palsu yang deterministik), di-ingest dua kali.
> - E2: `SinglePropertyExactMatchResolver` dan semantik `apoc.refactor.mergeNodes` (survivor, properti, relasi, self-loop, multi-label).
> - E3: bug `_consolidate_sets` sampai ke tahap merge di DB.
> - E4: `FuzzyMatchResolver` dan `SpaCySemanticMatchResolver` asli pada 12 entity lokasi, diulang 3 kali.
> - E5: skor `compute_similarity` asli, cabang `WRatio`, vocab spaCy, dan throughput.
>
> Versi sebelumnya memuat beberapa klaim yang ternyata keliru setelah diuji di DB. Klaim tersebut dikoreksi di tempatnya dan dirangkum di §15.
>
> Catatan: hybrid_rag saat ini hanya memakai `GraphSchema` dan `GraphPruning` dari package ini ([ontology_schema.py](../../hybrid_rag/ingestion/knowledge_graph/ontology_schema.py)), **bukan** resolver-nya.

---

## 1. Gambaran besar

### 1.1 Posisi resolver dalam `SimpleKGPipeline`

Komponen dan koneksinya ([`_get_connections`](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/experimental/pipeline/config/template_pipeline/simple_kg_builder.py#L325)):

```
file_loader → splitter → chunk_embedder → extractor → pruner → writer → resolver
                              schema ───────┘
```

- Resolver adalah komponen **terakhir**. Ia dipasang hanya jika `perform_entity_resolution=True` (**default**), dengan koneksi `writer → resolver` **tanpa input data** (`input_config={}`, [L402-408](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/experimental/pipeline/config/template_pipeline/simple_kg_builder.py#L402)). Resolver tidak tahu node mana yang baru ditulis. Ia bekerja langsung pada isi database.
- Resolver default adalah `SinglePropertyExactMatchResolver(driver, neo4j_database)` **tanpa `filter_query`** ([`_get_resolver` L317](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/experimental/pipeline/config/template_pipeline/simple_kg_builder.py#L317)). Akibatnya setiap run pipeline me-resolve **seluruh** node `__Entity__` di database, termasuk entity dari dokumen dan run sebelumnya. Terverifikasi (E1): pada ingest kedua, `number_of_nodes_to_resolve` = 17, yaitu 7 node lama ditambah 10 node baru. `SimpleKGPipeline` tidak menyediakan parameter untuk mengganti resolver atau `filter_query`. Pilihannya hanya on/off.
- Resolver fuzzy dan semantic **tidak** dipakai pipeline secara default. User harus memanggilnya sendiri setelah pipeline selesai.

### 1.2 Paradigma: "tulis dulu, satukan belakangan"

Desain package ini sengaja membuat **setiap penyebutan entity di setiap chunk menjadi node tersendiri**, lalu menyerahkan penyatuan sepenuhnya ke resolver post-hoc. Jadi resolver bukan fitur tambahan. Tanpa resolver, graph hasil pipeline terfragmentasi per chunk. Alasannya ada di tahap hulu (§2).

---

## 2. Tahap hulu yang menentukan perilaku merge

### 2.1 Extraction: ID lokal LLM, di-prefix `chunk_id`

1. LLM diminta mengembalikan `{"nodes":[{"id","label","properties"}], "relationships":[...]}`. Prompt hanya meminta: *"Assign a unique ID (string) to each node, and reuse it to define relationships"* ([prompts.py `ERExtractionTemplate` L177](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/generation/prompts.py#L177)). ID itu lokal untuk satu respons, biasanya `"0"`, `"1"`, dan seterusnya. Nama entity ada di `properties.name`, yang **tidak wajib** ada walaupun `name` didefinisikan di schema. Pruning baru membuang node tanpa `name` jika ada constraint `EXISTENCE`/`KEY` pada `name` ([`_check_required_properties`](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/graph_pruning.py#L476)). Terverifikasi (E1): node `Location` tanpa `name` tetap ditulis.
2. Setelah itu [`update_ids` L142](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/entity_relation_extractor.py#L142) mengubah setiap ID menjadi `f"{chunk.chunk_id}:{node.id}"`, begitu juga `start_node_id`/`end_node_id` relasi.
3. `chunk_id` adalah `uid = uuid4()` ([types.py L146-149](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/types.py#L146)). Akibatnya ID node unik **per chunk per run**. "Kota Malang" yang muncul di 10 chunk menjadi 10 node, dan meng-ingest ulang dokumen yang sama menggandakan semuanya.

Konsekuensinya, identitas entity **tidak pernah** ditentukan saat ekstraksi. Satu-satunya informasi identitas yang tersisa adalah nilai properti (umumnya `name`) dan label.

### 2.2 Lexical graph: provenance sebagai relasi

Jika `create_lexical_graph=True` (default), [`process_chunk_extracted_entities` L186](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/lexical_graph.py#L186) menambahkan relasi `(entity)-[:FROM_CHUNK]->(chunk)` untuk setiap node. Node `Chunk`/`Document` tidak diberi label `__Entity__`, sehingga tidak pernah ikut di-resolve.

Ini penting untuk merge: saat node-node digabung, relasi `FROM_CHUNK` mereka ikut dipindah ke node survivor. **Provenance tetap utuh lewat struktur graph**, walaupun properti node bisa hilang (§8).

### 2.3 Pruning: validasi schema, bukan identitas

[`GraphPruning`](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/graph_pruning.py) membuang node/relasi/properti yang tidak sesuai schema (label tidak dikenal, properti wajib hilang, pola relasi tidak valid). Tidak ada normalisasi nama atau deduplikasi di tahap ini.

### 2.4 Writer: selalu `CREATE`

[`Neo4jWriter.run` L298](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/kg_writer.py#L298):

1. `_db_setup`: index pada `(:__KGBuilder__).__tmp_internal_id`.
2. Node ditulis per batch 1000 dengan [`upsert_node_query` L85](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/neo4j_queries.py#L85):
   ```cypher
   UNWIND $rows AS row
   CREATE (n:__KGBuilder__ {__tmp_internal_id: row.id})
   SET n += row.properties
   SET n:$(row.labels)          -- atau apoc.create.addLabels untuk Neo4j < 5.24
   ```
   Label yang dipasang adalah `[node.label, "__Entity__"]` untuk non-lexical node ([`_nodes_to_rows` L234](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/kg_writer.py#L234)). Meskipun bernama "upsert", query ini **selalu `CREATE`**, bukan `MERGE`.
3. Relasi ditulis dengan [`upsert_relationship_query` L117](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/neo4j_queries.py#L117): `MATCH` kedua ujung lewat `__tmp_internal_id`, lalu `apoc.merge.relationship(start, type, {}, props, end, props)`. Hasilnya hanya satu relasi per `(start, type, end)`, tetapi karena ujung-ujungnya adalah node per chunk, ini tidak menyatukan apa pun lintas chunk.
4. `_db_cleaning`: `__tmp_internal_id` di-set `NULL` (`IN TRANSACTIONS ... ON ERROR CONTINUE`).
5. Constraint di `GraphSchema` (`UNIQUENESS`/`KEY`) **tidak** diterjemahkan menjadi `MERGE` atau constraint database oleh `Neo4jWriter`. Constraint itu hanya dipakai `ParquetWriter` untuk metadata kolom.

**Keadaan graph saat resolver mulai:** satu node per (chunk × entity yang disebut), masing-masing dengan `FROM_CHUNK` ke chunk asalnya.

**Terverifikasi end-to-end (E1)**, 3 chunk, ingest teks yang sama dua kali:

| | Setelah ingest #1 | Setelah ingest #2 |
|---|---|---|
| Chunk / Document | 3 / 1 | 6 / 2 (tergandakan, juga saat `from_file=False`) |
| "Kota Malang" (Location) | 1 node, `FROM_CHUNK` ke 3 chunk, `description='desc-A'` | 1 node, 6 `FROM_CHUNK`. `desc-B`/`desc-C` hilang. |
| "Malang Kota", "kota malang", "Kota  Malang" | 3 node terpisah | tetap 3 node terpisah |
| "Kota Malang" (Organization) | terpisah dari Location | terpisah |
| Node tanpa `name` | 1 | 2 (tergandakan) |
| Statistik resolver | `(10, 6)` | `(17, 6)` |

---

## 3. Kerangka resolver

```python
class EntityResolver(Component):           # resolver.py L53
    def __init__(self, driver, filter_query: Optional[str] = None)
    async def run(self) -> ResolutionStats  # number_of_nodes_to_resolve, number_of_created_nodes
```

- Semua resolver dimulai dari `MATCH (entity:__Entity__)` lalu menambahkan `filter_query` **sebagai string Cypher mentah** (contoh: `"WHERE entity.graph_id = 'doc-42'"`). Ini satu-satunya cara membatasi lingkup, misalnya hanya entity dari dokumen terbaru. Tidak ada parameterisasi. Package juga tidak menandai entity yang sudah di-resolve. Contoh `WHERE NOT entity:Resolved` di panduan resmi mengandaikan user memasang label itu sendiri.
- **Contoh `filter_query` kedua di panduan resmi tidak berfungsi.** `"WHERE NOT EXISTS((entity)-[:FROM_DOCUMENT]->(:OldDocument))"` ([user_guide_kg_builder.rst](../../neo4j-graphrag-python/docs/source/user_guide_kg_builder.rst)) tidak pernah cocok, karena entity hanya punya `FROM_CHUNK`, sedangkan `FROM_DOCUMENT` keluar dari Chunk. Terverifikasi (E7): entity dari `OldDocument` tetap ikut digabung. Pola yang benar: `WHERE NOT EXISTS((entity)-[:FROM_CHUNK]->()-[:FROM_DOCUMENT]->(:OldDocument))`.
- Pengelompokan **per label domain**: label `__Entity__` dan `__KGBuilder__` diabaikan. Pada graph keluaran writer bawaan (satu label domain per node), Person dan Organization dengan nama sama tidak pernah digabung (E1). Pengecualiannya node multi-label (§4).
- Merge dieksekusi oleh **APOC** (`apoc.refactor.mergeNodes`), jadi APOC wajib terpasang.
- `run()` bersifat `async`, tetapi memakai driver **sinkron** (`driver.execute_query`), sehingga event loop terblokir selama query berjalan.

Hierarki kelas:

```
EntityResolver
├── SinglePropertyExactMatchResolver          (Cypher murni, default pipeline)
└── BasePropertySimilarityResolver (abstract) (Python pairwise + APOC)
    ├── FuzzyMatchResolver                    (rapidfuzz WRatio)
    └── SpaCySemanticMatchResolver            (spaCy static vectors, cosine)
```

---

## 4. `SinglePropertyExactMatchResolver` — alur langkah demi langkah

Kode: [resolver.py L73-166](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L73). Parameter: `resolve_property="name"`, `filter_query`, `neo4j_database`.

**Langkah 1. Hitung lingkup** ([L120-132](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L120))
```cypher
MATCH (entity:__Entity__) <filter_query> RETURN count(entity) AS c
```
Jika 0, langsung return.

**Langkah 2. Satu query yang mengelompokkan dan merge** ([L133-158](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L133))
```cypher
MATCH (entity:__Entity__) <filter_query>
WITH entity, entity.name AS prop
WITH entity, prop WHERE prop IS NOT NULL               -- node tanpa name diabaikan
UNWIND labels(entity) AS lab
WITH lab, prop, entity WHERE NOT lab IN ['__Entity__','__KGBuilder__']
WITH prop, lab, collect(entity) AS entities             -- grup = (nilai persis, label)
CALL apoc.refactor.mergeNodes(entities, {properties:'discard', mergeRels:true})
YIELD node
RETURN count(node) AS c
```

**Semantik yang perlu dipahami:**

| Aspek | Perilaku |
|---|---|
| Kunci grup | `(nilai properti persis, label)`. **Tanpa normalisasi apa pun**: case-sensitive, spasi, tanda baca, dan diakritik semuanya berpengaruh. "Kota Malang" ≠ "kota malang" ≠ "Kota  Malang". |
| Node tanpa `name` | Tidak pernah di-resolve dan tetap sebagai node per chunk. |
| Grup berukuran 1 | Tetap dikirim ke `mergeNodes` (no-op). |
| Statistik | `count(node)` menghitung **jumlah grup, termasuk grup tunggal**, bukan jumlah node yang digabung. `number_of_created_nodes` = jumlah entity unik setelah resolusi. |
| Transaksi | **Satu transaksi** untuk seluruh lingkup. Tanpa `filter_query`, berarti seluruh database dalam satu transaksi, yang bisa membebani memori pada graph besar. |
| Survivor | Node **pertama** di `collect(entity)`, yaitu urutan **scan fisik** `NodeByLabelScan`. Urutan ini **bukan** urutan pembuatan setelah ada penghapusan, karena Neo4j memakai ulang slot ID yang dihapus. Contoh (E3b): node dibuat berurutan 1,3,5,6,7,8 lalu di-scan sebagai 3,8,5,1,6,7. Resolver sendiri menghapus node, jadi pada DB yang sudah berjalan, survivor tidak berkaitan dengan isi maupun kualitas nama. |
| Node multi-label | **Pemisahan per label bisa ditembus.** Node `:Location:Organization {name:'Y'}` masuk ke grup `(Y, Location)` dan `(Y, Organization)`. Terverifikasi (E2): `Location 'Y'`, `Organization 'Y'`, dan node multi-label itu **runtuh menjadi satu node** berlabel `Location+Organization`, tanpa error. Writer bawaan hanya memasang satu label domain, jadi risiko ini muncul pada graph yang diperkaya dari sumber lain. |

**Sifat yang diinginkan:** query ini **idempoten** (E2: run kedua tidak mengubah apa pun) dan berbiaya O(N) dengan agregasi native Cypher (`NodeByLabelScan` → `EagerAggregation`). Karena writer selalu `CREATE`, resolver inilah yang membuat re-ingest tidak menggandakan entity bernama sama (meski node Chunk/Document tetap tergandakan, §2.4).

---

## 5. `BasePropertySimilarityResolver` — alur langkah demi langkah

Kode: [resolver.py L177-311](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L177). Parameter: `resolve_properties=["name"]`, `similarity_threshold=0.8`, `filter_query`, `neo4j_database`. Subclass hanya mengimplementasikan `compute_similarity(text_a, text_b) -> float [0,1]`.

**Langkah 1. Muat semua entity per label ke Python** ([L235-244](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L235))
```cypher
MATCH (entity:__Entity__) <filter_query>
UNWIND labels(entity) AS lab
WITH lab, entity WHERE NOT lab IN ['__Entity__','__KGBuilder__']
WITH lab, collect({id: elementId(entity), name: entity.name, ...}) AS labelCluster
RETURN lab, labelCluster
```
Satu baris per label berisi **seluruh** entity berlabel itu, dimuat sekaligus ke memori.

**Langkah 2. Bentuk teks per node** ([L253-262](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L253))
Nilai `resolve_properties` yang tidak kosong digabung dengan spasi (`"Kota Malang Kota di Jawa Timur ..."` jika `["name","description"]`). Node dengan teks kosong dilewati.

**Langkah 3. Skor semua pasangan** ([L265-269](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L265))
```python
for (id1, t1), (id2, t2) in combinations(node_texts.items(), 2):
    if self.compute_similarity(t1, t2) >= self.similarity_threshold:
        pairs_to_merge.append({id1, id2})
```
- **O(n²) per label**, tanpa blocking, tanpa batas kandidat, Python murni dan sinkron.
- Keputusan berupa **satu threshold global** untuk semua label. Tidak ada zona "review", tidak ada veto.

**Langkah 4. Konsolidasi pasangan menjadi set** ([`_consolidate_sets` L298](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L298))
```python
for pair in pairs:
    for cons in consolidated:
        if pair & cons: cons.update(pair); break   # gabung ke set PERTAMA yang beririsan
    else: consolidated.append(set(pair))
```
- Maksudnya adalah **closure transitif** (A~B dan B~C membuat {A,B,C}), yang rawan *chain collapse*: Malang ~ Kota Malang ~ Pemerintah Kota Malang bisa menjadi satu node. Pada simulasi §10.5, 12 entity lokasi berbeda runtuh menjadi 3 node.
- **Bug (terverifikasi):** implementasi hanya satu lintasan dan berhenti di set pertama yang beririsan, sehingga dua set yang kemudian terhubung tidak pernah disatukan. Input `{a,b},{c,d},{b,c}` menghasilkan `[{a,b,c}, {c,d}]`, artinya node `c` ada di **dua** set (§10.3). Bug ini tercapai pada data nyata. Urutan pasangan mengikuti `combinations()` atas urutan scan DB, jadi cukup ada ≥ 5 node dengan pola seperti `n1~n5, n2~n3, n3~n5`. Pada 12 entity lokasi, bug ini muncul atau tidak tergantung urutan scan (§10.5).
- Tidak ada unit test maupun e2e test untuk kasus set yang beririsan. E2E test hanya menguji exact resolver ([test_entity_resolver_component_e2e.py](../../neo4j-graphrag-python/tests/e2e/components_e2e/test_entity_resolver_component_e2e.py)).

**Langkah 5. Merge per set, satu query per set** ([L276-289](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L276))
```cypher
MATCH (n) WHERE elementId(n) IN $ids
WITH collect(n) AS nodes
CALL apoc.refactor.mergeNodes(nodes, {properties:'discard', mergeRels:true})
YIELD node RETURN elementId(node)
```
- **Koreksi:** survivor **tidak** ditentukan oleh urutan iterasi `set` Python. Rencana query-nya adalah `AllNodesScan → Filter → EagerAggregation`, sehingga `collect(n)` selalu mengikuti urutan scan fisik, apa pun urutan `$ids` (E3b: 5 permutasi `$ids` menghasilkan urutan `collect` yang sama). Dalam semua run E3, survivor = anggota set yang paling awal di urutan scan. `PYTHONHASHSEED` tidak berpengaruh. Hasilnya deterministik untuk satu keadaan fisik DB, tetapi berubah begitu keadaan itu berubah (ID reuse setelah delete, re-ingest, atau resolver sebelumnya). E4: input identik menghasilkan survivor "Kabupaten Malang", "Kota Malang", dan "Malang"/"Malang Kota" pada run yang berbeda.
- **Temuan baru: setiap set memicu full scan seluruh DB.** `MATCH (n) WHERE elementId(n) IN $ids` tanpa label direncanakan sebagai `AllNodesScan` (Neo4j 5.26). Bentuk `elementId(n) = x` (atau `UNWIND $ids ... WHERE elementId(n) = i`) memakai `NodeByElementIdSeek`. Biaya tahap merge menjadi O(jumlah set × total node DB), termasuk node Chunk/Document.
- Setiap set dieksekusi sebagai transaksi auto-commit terpisah. Prosesnya **tidak atomik**: kegagalan di tengah meninggalkan graph setengah ter-resolve.
- ID berasal dari snapshot Langkah 1. Akibat bug Langkah 4, node yang ada di dua set mungkin sudah terhapus saat set kedua diproses. Terverifikasi di DB (E3, pola `n1~n5, n2~n3, n3~n5`, hasil benar adalah 2 node):
  - Jika survivor set pertama adalah `n3` (node yang ada di dua set), set kedua tetap menemukan `n3` dan hasilnya kebetulan benar (2 node).
  - Jika survivornya `n1`/`n5`, `n3` sudah terhapus. Set kedua hanya menemukan `n2`, `mergeNodes([n2])` menjadi no-op, dan hasilnya **3 node (under-merge)**. Statistik tetap melaporkan 2 merge.

---

## 6. `FuzzyMatchResolver` — anatomi skor

Kode: [resolver.py L434-470](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L434).
```python
fuzz.WRatio(text_a, text_b, processor=utils.default_process) / 100.0
```

- `default_process`: lowercase, karakter non-alfanumerik diganti spasi, trim. Ini satu-satunya normalisasi di seluruh package.
- `WRatio` adalah **maksimum berbobot** dari beberapa skor. Rinciannya diverifikasi dari source rapidfuzz 3.14.6 (`fuzz_py.WRatio`):
  - `ratio` selalu dihitung. Ini **similarity Indel ternormalisasi** (hanya insert/delete), **bukan** Levenshtein seperti tertulis di panduan resmi package.
  - Jika rasio panjang **< 1,5**: `max(ratio, max(token_sort, token_set) × 0,95)`.
  - Jika rasio panjang **≥ 1,5**: `max(ratio, partial_ratio × s, partial_token_ratio × 0,95 × s)`, dengan `s = 0,9` jika rasio ≤ 8 dan `s = 0,6` jika > 8. Nilai **0,855** yang sering muncul (§10.5) adalah `partial_token_ratio` 100 × 0,95 × 0,9.

Konsekuensi langsung dengan threshold default 0,8:

| Pola | Skor tipikal | Akibat |
|---|---|---|
| Urutan token tertukar ("Kota Malang"/"Malang Kota") | 0,95 (token_sort × 0,95) | merge ✓ |
| Satu nama **substring** nama lain, rasio panjang 1,5–8 ("Malang"/"Kota Malang", "Indonesia"/"Bank Indonesia") | 0,90 (partial × 0,9) | **merge ✗** |
| Himpunan token satu nama ⊂ nama lain ("Kota Malang"/"Pemerintah Kota Malang") | 0,90–0,95 | **merge ✗** (jika labelnya sama) |
| Beda satu angka ("Pasal 9"/"Pasal 10", "UU No. 24"/"UU No. 25") | 0,95 (`ratio` biasa) | **merge ✗** |
| Awalan panjang yang sama ("Kementerian Keuangan"/"Kementerian Perdagangan") | 0,84 (`ratio` biasa) | **merge ✗** |
| Akronim vs bentuk panjang ("ICCN"/"Indonesia Creative Cities Networks (ICCN)", rasio panjang 10,0 setelah `default_process`) | 0,60 (partial × 0,6) | tidak merge |

Dengan `resolve_properties=["name","description"]`, deskripsi yang mirip mengangkat skor pasangan berbeda. "Kota Malang"+deskripsi vs "Kabupaten Malang"+deskripsi mencapai 0,77–0,79 (bergantung teks deskripsi), tipis di bawah threshold.

---

## 7. `SpaCySemanticMatchResolver` — anatomi skor

Kode: [resolver.py L314-431](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L314).

Diuji dengan model default yang sebenarnya: **`en_core_web_lg` 3.8.0**, 342.918 static vectors × 300 dimensi.

**Cara skor dihitung**

1. `compute_similarity` memanggil `self.nlp(text).vector` untuk kedua teks ([L370-382](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L370)).
2. `Doc.vector` adalah **rata-rata vektor semua token, termasuk token OOV yang vektornya nol**. Ini terverifikasi: `nlp("Kota Malang").vector == mean(token.vector)`.
3. Cosine dihitung di [`_cosine_similarity` L385](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L385). Jika salah satu norma 0, hasilnya `0.0`.

**Tiga konsekuensi matematis yang menentukan perilakunya**

- **Token OOV tidak terlihat oleh cosine.** Token OOV menyumbang vektor nol, yang hanya mengecilkan panjang rata-rata tanpa mengubah arahnya. Cosine tidak bergantung pada panjang vektor, jadi dua nama yang **hanya berbeda pada token OOV mendapat skor 1,000**. Contoh: "Kota Malang" vs "Pemerintah Kota Malang", "Program KaTa Kreatif" vs "Program Kreatif", "Jawa Timur" vs "Provinsi Jawa Timur".
- **Nama yang seluruh tokennya OOV menghasilkan vektor nol, sehingga skornya 0,000 terhadap apa pun.** Akronim (`ICCN`, `Kemenparekraf`, `Bekraf`) dan nama dengan kosakata Indonesia formal (`Kementerian`, `Kabupaten`, `Pasal`) tidak pernah cocok dengan bentuk panjangnya. Akibat sampingnya, "Kementerian Keuangan" vs "Kementerian Perdagangan" juga 0,000. Pasangan ini aman bukan karena modelnya membedakan, melainkan karena modelnya tidak mengenali kata-katanya sama sekali.
- **Yang tersisa hanya token dalam vocab Inggris, dan token itu mendominasi.** `Kota`, `Malang`, `Jawa`, `Timur`, `Barat`, `Tahun`, `Indonesia`, dan angka (`9`, `10`, `24`, `25`) punya vektor. Nama kota lain yang ada di vocab (Surabaya, Bandung, Yogyakarta, Batu) dan arah mata angin bertetangga dekat di ruang embedding, sehingga "Kota Malang"↔"Kota Surabaya" (0,886) dan "Jawa Timur"↔"Jawa Barat" (0,874) melewati threshold. Antar pasangan "Kota X" skornya 0,795–0,904. **Koreksi:** tidak semuanya ≥ 0,81, karena "Kota Batu"↔"Kota Yogyakarta" = 0,795. Hanya 9 dari 10 pasangan yang lolos langsung, dan sisanya tersambung lewat closure transitif. Vektor angka juga saling dekat: "Pasal 9"↔"Pasal 10" mendapat 0,993.

Cakupan vocab pada token korpus hybrid_rag (`token.has_vector`, E5):

| Ada vektor | OOV (vektor nol, tidak terlihat) |
|---|---|
| Kota/kota, Malang/malang, Tahun, Jawa, Timur, Barat, Indonesia, Creative, Program, UU, No., 9, 10, 24, 25 | ICCN, Kemenparekraf, Bekraf, Kabupaten, Kementerian, Pasal, Pemerintah, Provinsi, KaTa |

Contoh paling jelas: skor "Kota Malang"↔"Kabupaten Malang" (0,889) **persis sama** dengan "Kota Malang"↔"Malang" (0,889). Kata "Kabupaten" benar-benar tidak terlihat oleh model.

**Biaya operasional (diukur ulang, E5/E6, WSL2)**
- Memuat model butuh 2,8–3,5 detik dengan RSS ±740 MB per proses. Model terpasang berukuran 425 MB.
- Model diunduh otomatis saat konstruktor dipanggil (`auto_download_spacy_model=True`), sehingga ada efek samping jaringan dan disk saat inisialisasi.
- `self.nlp(text)` menjalankan **seluruh pipeline** (`tok2vec, tagger, parser, attribute_ruler, lemmatizer, ner`) hanya untuk mengambil `.vector`. Embedding baru ±120 teks/detik (10.000 entity ≈ 85 detik), padahal tokenizer saja sudah cukup.
- Skoring dengan cache embedding: ±130.000 pasangan/detik. 10.000 entity dalam satu label (±50 juta pasangan) ≈ **6–7 menit** skoring, ditambah ±1,5 menit embedding. **Koreksi:** angka sebelumnya (16.300 pasangan/detik, ±51 menit) sekitar 8× terlalu pesimis. Sebagai pembanding, Fuzzy ±480.000 pasangan/detik (≈ 1,7 menit untuk 10.000 entity). Biaya merge APOC belum termasuk (lihat full scan per set, §5).

---

## 8. Semantik merge: `apoc.refactor.mergeNodes`

Semua resolver memakai konfigurasi yang sama: `{properties: 'discard', mergeRels: true}`.

| Aspek | Perilaku | Dampak |
|---|---|---|
| Node survivor | Node **pertama** di list. Node lain dihapus. | Kedua jenis resolver: list berasal dari `collect` dengan urutan scan fisik DB (§4, §5), bukan dari urutan `set` Python. |
| Properti (`discard`) | Nilai survivor dipertahankan. Properti yang tidak dimiliki survivor diambil dari node lain (E2: survivor `{name:'Z', description:'d1', only1}` mendapat `only2` dari node kedua). | `name` dan `description` node lain **hilang** (E1: `desc-B`, `desc-C`). Tidak ada penggabungan deskripsi, tidak ada ringkasan. |
| Label | Label semua node digabung ke survivor. | Aman jika grup satu label domain. Node multi-label membuat grup lintas label runtuh (§4). |
| Relasi (`mergeRels: true`) | Semua relasi dipindah ke survivor. Relasi dengan tipe dan node lawan yang sama digabung menjadi satu, dengan **gabungan key properti**. Untuk key yang konflik, nilainya **tidak** mengikuti survivor (E7: survivor `first`, tetapi `k` relasi hasilnya `'second'`). | `FROM_CHUNK` ke chunk yang sama digabung, dan provenance ke semua chunk asal **tetap ada** (E1). Properti relasi domain (mis. `since`) bisa berasal dari node yang dibuang. |
| Relasi antar anggota grup | Jika node yang digabung saling berelasi (A-[:X]->B), relasi itu menjadi self-loop pada survivor. Package tidak mengubah opsi APOC terkait. Terverifikasi (E2): `Pasal 9 -[:MENGACU]-> Pasal 9`. | Self-loop palsu. Terutama terjadi pada false merge seperti "Pasal 9"-[:MENGACU]->"Pasal 10". |
| Jejak merge | **Tidak ada.** Tidak ada `aliases`, `merged_from`, log keputusan, maupun skor. | Tidak bisa diaudit atau di-undo. Nama varian hilang dari graph, sehingga pencarian "Malang Kota" tidak lagi menemukan node. |

---

## 9. Kustomisasi dan extension point

| Knob | Fungsi | Batasan |
|---|---|---|
| `perform_entity_resolution` | Menyalakan atau mematikan resolver exact di pipeline. | Hanya on/off. |
| `filter_query` | Membatasi lingkup lewat klausa Cypher mentah. | Harus ditulis manual, tidak diparameterisasi, tidak dihubungkan otomatis ke dokumen yang baru di-ingest, dan tidak bisa dipasang lewat `SimpleKGPipeline`. Salah satu contoh resmi tidak berpengaruh apa pun (§3). |
| `resolve_property` / `resolve_properties` | Properti yang dibandingkan. | Exact hanya satu properti. Similarity menggabungkan teks, bukan membandingkan per field. |
| `similarity_threshold` | Ambang merge. | Satu angka global, tanpa zona review dan tanpa per-label. |
| Subclass `compute_similarity` | Logika skor sendiri, misalnya alias dictionary yang mengembalikan 1,0. | Pemuatan per label, O(n²), konsolidasi transitif yang bug, dan merge `discard` tetap diwarisi. |
| Rangkaian resolver manual | Misalnya exact lalu fuzzy dengan `filter_query` berbeda per label. | Harus diorkestrasi sendiri oleh user. |

Package ini **tidak** menyediakan alias dictionary, daftar never-merge, conflict properties, guard numerik, LLM adjudication, mode review/dry-run, atau resolusi saat ingest.

---

## 10. Uji empiris pada kasus target

### 10.1 `SinglePropertyExactMatchResolver`
Perbandingannya string persis, jadi hanya nama yang **identik byte-per-byte** (dan berlabel sama) yang digabung. Semua kasus target (ICCN, Kota Malang/Malang Kota, bahkan "Kota Malang"/"kota malang" dan "Kota  Malang" dengan spasi ganda) **tidak** tergabung. Ini terverifikasi lewat pipeline penuh (E1).

### 10.2 `FuzzyMatchResolver` (kode `compute_similarity` asli, threshold default 0,8)

| A | B | WRatio | Keputusan | Seharusnya |
|---|---|---|---|---|
| ICCN | Indonesia Creative Cities Networks (ICCN) | 0,60 | tidak | **merge** |
| ICCN | Indonesia Creative Cities Networks | 0,40 | tidak | **merge** |
| Kota Malang | Malang Kota | 0,95 | merge | merge ✓ |
| Kota Malang | kota malang | 1,00 | merge | merge ✓ |
| Kemenparekraf | Kementerian Pariwisata Dan Ekonomi Kreatif | 0,63 | tidak | **merge** |
| Kota Malang | Kabupaten Malang | 0,67 | tidak | tidak ✓ |
| Kota Malang | Pemerintah Kota Malang | 0,90 | **merge** | tidak (jika labelnya sama) |
| Malang | Kota Malang | 0,90 | **merge** | ambigu (kota/kabupaten) |
| UU No. 24 Tahun 2019 - Pasal 9 | UU No. 24 Tahun 2019 - Pasal 10 | 0,95 | **merge** | tidak |
| UU No. 24 Tahun 2019 | UU No. 25 Tahun 2019 | 0,95 | **merge** | tidak |
| Kementerian Keuangan | Kementerian Perdagangan | 0,84 | **merge** | tidak |
| Bank Indonesia | Indonesia | 0,90 | **merge** | tidak |
| Program KaTa Kreatif | Program Kreatif | 0,95 | **merge** | tidak |
| Jawa Timur | Jawa Barat | 0,60 | tidak | tidak ✓ |
| Jawa Timur | Provinsi Jawa Timur | 0,90 | **merge** | tidak (jika labelnya sama) |
| *name+description* Kota Malang | *name+description* Kabupaten Malang | 0,77–0,79 | tidak (tipis) | tidak |

### 10.3 `_consolidate_sets` (kode asli)

| Input pasangan | Output | Seharusnya |
|---|---|---|
| `{a,b}, {c,d}, {b,c}` | `[{a,b,c}, {c,d}]`, **`c` muncul dua kali** | `[{a,b,c,d}]` |
| `{A,B}, {B,C}, {C,D}` | `[{A,B,C,D}]` | Transitif penuh (chain collapse jika B~C salah) |

Efeknya di DB (E3, 10 run, pola `n1~n5, n2~n3, n3~n5`, ideal 2 node): 8 run menghasilkan 2 node, 2 run menghasilkan **3 node** (`n2` tidak tergabung). Yang menentukan adalah urutan scan DB (survivor set pertama), bukan seed hash Python.

### 10.4 `SpaCySemanticMatchResolver`

Diuji dengan model default **`en_core_web_lg` 3.8.0** dan method `compute_similarity` asli package, threshold default 0,8:

| A | B | cosine | Keputusan | Seharusnya |
|---|---|---|---|---|
| ICCN | Indonesia Creative Cities Networks (ICCN) | 0,000 | tidak | **merge** |
| ICCN | Indonesia Creative Cities Networks | 0,000 | tidak | **merge** |
| Kemenparekraf | Kementerian Pariwisata Dan Ekonomi Kreatif | 0,000 | tidak | **merge** |
| Badan Ekonomi Kreatif | Bekraf | 0,000 | tidak | **merge** |
| Kota Malang | Malang Kota | 1,000 | merge | merge ✓ |
| Kota Malang | kota malang | 1,000 | merge | merge ✓ |
| Kota Malang | Kabupaten Malang | 0,889 | **merge** | tidak |
| Kota Malang | Pemerintah Kota Malang | 1,000 | **merge** | tidak (jika labelnya sama) |
| Malang | Kota Malang | 0,889 | **merge** | ambigu |
| Kota Malang | Kota Surabaya | 0,886 | **merge** | tidak |
| UU No. 24 Tahun 2019 - Pasal 9 | UU No. 24 Tahun 2019 - Pasal 10 | 0,993 | **merge** | tidak |
| UU No. 24 Tahun 2019 | UU No. 25 Tahun 2019 | 0,989 | **merge** | tidak |
| Jawa Timur | Jawa Barat | 0,874 | **merge** | tidak |
| Jawa Timur | Provinsi Jawa Timur | 1,000 | **merge** | tidak (jika labelnya sama) |
| Kota Batu | Kota Yogyakarta | 0,795 | tidak (tetapi tergabung lewat rantai, §10.5) | tidak |
| Program KaTa Kreatif | Program Kreatif | 1,000 | **merge** | tidak |
| Bank Indonesia | Indonesia | 0,800 (tepat di bawah 0,8) | tidak (nyaris) | tidak |
| Kementerian Keuangan | Kementerian Perdagangan | 0,000 | tidak | tidak ✓ (karena semua tokennya OOV) |
| *name+description* Kota Malang | *name+description* Kabupaten Malang | 0,924 | **merge** | tidak |

Menambahkan `description` ke `resolve_properties` justru **memperburuk** keadaan: deskripsi berbahasa Indonesia yang strukturnya mirip ("… di Jawa Timur yang dikenal sebagai …") menaikkan skor pasangan berbeda dari 0,889 menjadi 0,92–0,93.

### 10.5 Resolver penuh di Neo4j: efek closure transitif

**Revisi.** Versi sebelumnya hanya mensimulasikan Langkah 3–4 di Python dan menyimpulkan "12 → 3 node, cluster sama persis untuk kedua scorer". Kesimpulan itu keliru, dan dua cluster yang dicantumkannya (9 + 3 entity) sebenarnya berarti 2 node. Sekarang `FuzzyMatchResolver.run()` dan `SpaCySemanticMatchResolver.run()` asli dijalankan pada Neo4j, masing-masing 3 kali dengan DB yang diisi ulang. Setiap entity diberi Chunk sendiri, sehingga keanggotaan cluster bisa dilacak lewat `FROM_CHUNK` walaupun properti di-*discard*.

Input berupa 12 entity berlabel Location yang semuanya berbeda kecuali "Kota Malang"/"Malang Kota":

```
Kota Malang, Malang Kota, Kabupaten Malang, Malang, Pemerintah Kota Malang,
Kota Surabaya, Kota Batu, Kota Bandung, Kota Yogyakarta,
Jawa Timur, Jawa Barat, Provinsi Jawa Timur
```

| Scorer | Pasangan langsung ≥ 0,8 | Node akhir (3 run) | Cluster | Survivor |
|---|---|---|---|---|
| SpaCy `en_core_web_lg` | 30 dari 66 | **2, 2, 2** | {9 entity Malang/Kota X} + {Jawa Timur, Jawa Barat, Provinsi Jawa Timur} | "Malang", "Malang Kota", "Malang Kota" |
| Fuzzy `WRatio` | 13 dari 66 | **3, 3, 2** | {8 entity} + {3 Jawa} + 1 entity lolos (run 1: Kota Surabaya, run 2: Kota Yogyakarta), atau {9} + {3} | "Kabupaten Malang", "Kota Malang", "Kota Malang" |

Penyebabnya berbeda untuk kedua resolver:
- **SpaCy**: 9 dari 10 pasangan "Kota X" sudah ≥ 0,8 secara langsung (0,795–0,904), dan "Pemerintah Kota Malang"/"Kabupaten Malang" berbagi vektor dengan "Kota Malang" karena token OOV. Grafnya terhubung rapat, sehingga hasilnya stabil di 2 node.
- **Fuzzy**: skor langsung antar kota rendah (mis. "Kota Surabaya"↔"Kota Bandung" 0,56), tetapi jembatan `partial_token_ratio` × 0,855 meruntuhkannya lewat transitivitas: "Pemerintah Kota Malang"↔"Kota Surabaya"/"Kota Batu"/"Kota Bandung" dan "Kota Batu"↔"Kota Yogyakarta". Graf yang jarang ini memicu bug `_consolidate_sets`. Pada urutan scan tertentu, satu kota tertinggal di set yang beririsan dan lolos dari merge. Jadi **jumlah node akhir dan entity mana yang selamat bergantung pada urutan fisik DB**.

Hasilnya, 12 entity menjadi **2–3 node** padahal seharusnya 11. Delapan sampai sembilan entity berbeda (kota lain, kabupaten, pemerintah kota, dan provinsi) hilang ke dalam node lain dengan `properties:'discard'`, tanpa jejak. Karena merge memakai `mergeRels:true`, semua relasi mereka (misalnya program kota Bandung dan Surabaya) kini menempel pada satu node yang namanya bergantung pada urutan scan.

### 10.6 Ringkasan uji

- **Exact**: presisi sempurna, recall nol untuk semua variasi.
- **Fuzzy**: menangkap "Kota Malang"↔"Malang Kota" tanpa konfigurasi, tetapi pada threshold yang sama juga menggabungkan pasal berbeda, undang-undang berbeda, kementerian berbeda (awalan sama), dan nama yang hanya substring. Positif "Kota Malang"/"Malang Kota" (0,95) bernilai **sama** dengan negatif "Pasal 9"/"Pasal 10" (0,95), jadi tidak ada threshold tunggal yang memisahkan keduanya.
- **Semantic (`en_core_web_lg`)**: lebih buruk dari fuzzy untuk korpus Indonesia. Token OOV tidak terlihat (skor 1,000 untuk nama yang hanya berbeda di kata Indonesia), nama yang seluruhnya OOV tidak bisa dicocokkan (0,000), dan pasangan "Kota X" berskor 0,795–0,904.
- **Closure transitif** mengubah beberapa false pair menjadi keruntuhan massal. Dari 12 entity lokasi tersisa 2 node (spaCy) atau 2–3 node (fuzzy, berubah antar run).
- **Hasil fuzzy tidak stabil**: input identik menghasilkan jumlah node dan survivor yang berbeda, bergantung urutan scan DB.
- **ICCN, Kemenparekraf, dan Bekraf tidak tertangani oleh resolver mana pun.**

---

## 11. Daftar temuan (berdasarkan tingkat keparahan)

| # | Keparahan | Temuan | Lokasi |
|---|---|---|---|
| 1 | Kritis | Fuzzy/semantic tanpa guard numerik maupun discriminator, sehingga pasal, UU bernomor berbeda, dan (fuzzy) kementerian berbeda tergabung pada threshold default. | [L265-269](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L265), [L465-470](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L465) |
| 1b | Kritis | SpaCy: `Doc.vector` merata-ratakan token OOV sebagai vektor nol, sehingga kata Indonesia tidak terlihat oleh cosine. Nama yang hanya berbeda di token OOV mendapat 1,000, dan pasangan "Kota X" 0,795–0,904. Bersama closure transitif, 12 entity lokasi runtuh menjadi 2 node di Neo4j (§10.5). | [L370-394](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L370) |
| 2 | Kritis | Merge `properties:'discard'` tanpa jejak: nama varian dan deskripsi node lain hilang, tidak ada `aliases`, tidak bisa di-undo. | [L152-155](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L152), [L281](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L281) |
| 3 | Tinggi | `_consolidate_sets` tidak menyatukan set yang terhubung belakangan. Di DB ini menyebabkan **under-merge** yang bergantung urutan scan (E3: 2 dari 10 run; E4 fuzzy: 3, 3, 2 node). Tidak ada test untuk kasus ini. | [L298-311](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L298) |
| 4 | Tinggi | Survivor ditentukan **urutan scan fisik DB** (bukan urutan pembuatan, bukan `set` Python), sehingga nama canonical bergantung pada riwayat delete/ID reuse dan berubah antar keadaan DB. | [L145](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L145), [L279-280](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L279) |
| 5 | Tinggi | Closure transitif tanpa batas menyebabkan chain collapse lewat pasangan substring (WRatio `partial`/`partial_token` × 0,9). | [L272](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L272) |
| 6 | Tinggi | Exact match tanpa normalisasi apa pun (case, spasi, tanda baca). | [L135](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L135) |
| 7 | Sedang | Similarity O(n²) per label, memuat seluruh label ke memori, tanpa blocking. | [L240](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L240), [L266](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L266) |
| 7b | Sedang | **Baru:** query merge per set `MATCH (n) WHERE elementId(n) IN $ids` dieksekusi sebagai `AllNodesScan`, sehingga setiap set memindai seluruh DB (O(set × total node)). | [L279](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L279) |
| 8 | Sedang | Merge per set tidak atomik dan memakai elementId dari snapshot. | [L276-289](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L276) |
| 9 | Sedang | Pipeline default me-resolve seluruh database setiap run (tanpa `filter_query`, tidak bisa diatur dari `SimpleKGPipeline`), dalam satu transaksi. | [simple_kg_builder.py L317](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/experimental/pipeline/config/template_pipeline/simple_kg_builder.py#L317) |
| 9b | Sedang | **Baru:** contoh `filter_query` resmi `(entity)-[:FROM_DOCUMENT]->(:OldDocument)` tidak berpengaruh apa pun, karena entity hanya terhubung ke Chunk. | [user_guide_kg_builder.rst](../../neo4j-graphrag-python/docs/source/user_guide_kg_builder.rst) |
| 9c | Sedang | **Baru:** node multi-label menembus pemisahan per label, sehingga grup `(nama, LabelA)` dan `(nama, LabelB)` runtuh menjadi satu node. | [L140-145](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L140) |
| 10 | Sedang | Writer selalu `CREATE` dengan ID per chunk (uuid4), sehingga graph terfragmentasi penuh sebelum resolusi dan re-ingest menggandakan Chunk/Document. | [neo4j_queries.py L103](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/neo4j_queries.py#L103), [extractor L150](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/entity_relation_extractor.py#L150) |
| 11 | Rendah | Statistik menyesatkan. Exact: `number_of_created_nodes` menghitung grup termasuk grup tunggal. Similarity: menghitung set yang dieksekusi, termasuk set yang menyusut menjadi no-op. | [L157](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L157), [L289](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L289) |
| 12 | Rendah | `async run()` memakai driver sinkron, sehingga event loop terblokir. | [L124](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L124) |
| 13 | Rendah | SpaCy mengunduh model berbahasa Inggris (425 MB) secara otomatis di konstruktor, butuh RSS ±740 MB, dan menjalankan seluruh pipeline NLP (parser, NER) hanya untuk mengambil vektor. 10 rb entity per label butuh ±8 menit (embedding + skor). | [L380](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L380), [L397-431](../../.venv/lib/python3.12/site-packages/neo4j_graphrag/components/resolver.py#L397) |

## 12. Kekuatan

1. **Sederhana dan mudah dipahami**: seluruh mekanisme ada di 470 baris, dan exact resolver hanya satu query Cypher.
2. **Exact resolver native-Cypher**: agregasi di database, O(N), idempoten. Ini cara yang tepat untuk tahap "exact key" di level database.
3. **Pengelompokan wajib per label**: guard tipe yang murah dan efektif. Pada graph satu-label-domain, Location vs Organization tidak pernah tergabung (E1). Guard ini tidak berlaku untuk node multi-label (§4).
4. **Provenance lewat relasi `FROM_CHUNK`** yang ikut berpindah saat merge (`mergeRels`), sehingga sumber chunk tidak hilang walaupun properti di-*discard*.
5. **Extension point yang bersih** (`compute_similarity`) untuk mencoba skor lain tanpa menyentuh mekanisme merge.
6. **Tanpa LLM**: biaya hanya CPU dan database.

## 13. Penilaian

| Dimensi | Skor (1–5) | Catatan |
|---|---|---|
| Recall kasus target | 2 | Fuzzy dan spaCy menangkap token-swap. Akronim tidak tertangani resolver mana pun (spaCy: 0,000). |
| Precision / keamanan | 1 | Threshold tunggal tanpa guard, token OOV tidak terlihat (spaCy), closure transitif (12 lokasi menjadi 2–3 node di Neo4j), merge `discard` tanpa jejak. |
| Kustomisasi | 2 | `filter_query`, threshold, subclass skor. Tanpa alias, never-merge, maupun konflik. Resolver dan `filter_query` tidak bisa diatur dari `SimpleKGPipeline`. |
| Efisiensi | 3 | Exact: sangat efisien. Similarity: O(n²) Python (cukup cepat sampai ±10 rb entity per label), seluruh label di memori, dan merge per set full-scan DB. Tanpa biaya LLM. |
| Auditability | 1 | Hanya dua angka statistik, dan keduanya menyesatkan. Tidak ada log keputusan atau alias. |
| Determinisme | 2 | Survivor bergantung urutan scan fisik DB. Hasil fuzzy (jumlah node) berubah antar run pada input identik. |
| Kematangan kode | 3 | Library resmi dan rapi, tetapi resolver similarity masih naif (bug konsolidasi). |

---

## 14. Perbandingan (bagian terakhir)

Perbandingan dengan dua package yang dianalisis sebelumnya ([01](01-recon-graphrag.md), [02](02-docling-graph.md)):

| Aspek | neo4j-graphrag | recon_graphrag | docling-graph |
|---|---|---|---|
| Kapan berjalan | Post-hoc, seluruh DB, setiap run | Dedup exact saat ingest (`MERGE` pada UUID) + resolusi post-hoc atas seluruh graph setiap build | Saat konversi per dokumen (L3 hanya contract dense) + fusi offline |
| Identitas saat tulis | Tidak ada (ID per chunk, `CREATE`) | `uuid5(canonical_key)`, dengan `canonical_key` = ID `<label>:<normalized-name>` yang dinormalisasi **oleh LLM** | Hash kunci kanonik dari template |
| Normalisasi | Tidak ada (exact). `default_process` (fuzzy). | casefold, hapus tanda baca **dan spasi**, buang suffix badan usaha Inggris di posisi mana pun | NFKD, diakritik, artikel, posesif, token dipertahankan |
| Candidate generation | Semua pasangan per label | O(n²); blocking efektif hanya **huruf pertama sama + panjang mirip**; batas 20 kandidat menurut urutan load | Containment + guard |
| Keputusan | Satu threshold | `fuzz.ratio` ≥95 merge, ≥85 review. Hybrid: LLM per pasangan termasuk semua *rescue*; auto-merge opsional (confidence ≥ 0,95). | L4: proposer ∩ LLM. L3 (dense): LLM bebas dalam satu path, dibatasi veto. |
| Guard / veto | Hanya label | Label + `conflict_properties` (hanya hybrid). Tanpa guard numerik. | Label, digit signature, co-occurrence per batch/chunk, sibling, base ambigu, split konflik lintas dokumen |
| Clustering | Transitif (dan bug) | Star non-transitif (fuzzy); exact key (normalized) | Grup keep/merge tervalidasi |
| Alias kustom | Tidak | Ya, tetapi bisa tidak pernah konvergen ([01 §6.3](01-recon-graphrag.md)) | Tidak |
| Jejak merge | Tidak ada | `aliases` di node. `review_groups` tidak dikembalikan dan tidak disimpan lewat pipeline. | `merged_aliases`, decision file, lineage |
| LLM | Tidak | Ya (hybrid; default audit-only), ±17 panggilan per entity per build tanpa cache | Ya (id-space; L3 batch ≤100, L4 satu panggilan per graph) |
| Kota Malang ↔ Malang Kota | **Ya** (fuzzy 0,95; spaCy 1,000) | Tidak (huruf pertama beda; LLM oracle pun tidak melihatnya) | Deterministik hanya dense + `aggressive` (L2). Selain itu bergantung LLM. |
| ICCN ↔ bentuk panjang | Tidak (spaCy 0,000) | Tidak (gugur di filter panjang). Alias dict bisa, tetapi bisa tidak konvergen. | Kandidat hanya jika "(ICCN)" literal. Diveto bila sibling di list yang sama. |
| False merge Pasal 9/10, UU 24/25 | **Ya** (fuzzy 0,95; spaCy 0,99) | Lolos blocking dan masuk review (rasio ±93). Tergabung bila auto-merge aktif dan LLM menyetujui. | Tidak (digit guard) |
| False merge Kota X ↔ Kota Y / Jawa Timur ↔ Jawa Barat | **Ya** (spaCy langsung; fuzzy lewat rantai) | Tidak secara deterministik. Pasangan ini lolos blocking ke LLM rescue, dan tergabung bila auto-merge aktif dan LLM menyetujui. | Tidak |
| False merge pada strategi default | Tidak (exact resolver) | **Ya**: `normalized` menggabungkan "Lampiran 1.2"="Lampiran 12" dan "LP Cipinang"="Cipinang" tanpa review | Entity yang semua field ID-nya kosong digabung jadi satu |
| Keruntuhan cluster (12 entity lokasi, ideal 11 node) | Dijalankan di Neo4j (§10.5): spaCy 12 → 2 node, fuzzy 12 → 2–3 node (berubah antar run) | Disimulasikan dengan kode asli: normalized/fuzzy dan hybrid + LLM oracle tetap 12 node (Kota Malang↔Malang Kota terlewat). Hybrid + LLM "selalu ya" + auto-merge menghasilkan 12 → 8 node: 4 false merge **berpasangan** (mis. Kota Malang=Kota Surabaya, Kabupaten Malang=Kota Bandung), tanpa rantai. | Tidak disimulasikan. Tidak ada closure transitif otomatis: setiap merge adalah pasangan yang diusulkan proposer dan dikonfirmasi LLM, dan node yang sudah diserap dilewati. Rantai tetap mungkin lewat survivor, tetapi setiap mata rantainya tervalidasi sendiri. L2 tidak menghasilkan false merge pada set uji [02 §6.1](02-docling-graph.md). |

**Peringkat robustness:** docling-graph > recon_graphrag > neo4j-graphrag. neo4j-graphrag adalah satu-satunya yang menangkap token-swap tanpa konfigurasi, tetapi resolver similarity-nya (baik fuzzy maupun spaCy `en_core_web_lg`) menggabungkan entity berbeda secara massal pada data regulasi dan wilayah yang menjadi inti korpus hybrid_rag, tanpa jejak untuk memperbaikinya. Model semantic bawaannya berbahasa Inggris, sehingga kata Indonesia tidak terlihat oleh skornya.

**Peringkat efisiensi:** exact resolver neo4j-graphrag adalah tahap termurah (satu agregasi Cypher). Untuk similarity, docling-graph tetap paling efisien per keputusan berkualitas. neo4j-graphrag fuzzy murah karena tanpa LLM (±480 rb pasangan/detik) tetapi O(n²) per label, dan setiap set merge melakukan full scan DB. recon paling mahal: blocking O(n²) atas seluruh graph (37,8 detik untuk 4.000 entity satu label) dan ±17 panggilan LLM per entity per build pada hybrid, tanpa cache keputusan ([01 §6.4](01-recon-graphrag.md)).

**Yang layak diambil dari neo4j-graphrag:** pengelompokan exact-key langsung di Cypher sebagai tahap backfill, guard wajib per label, dan pola memindahkan provenance lewat relasi saat merge (dengan `properties` yang **tidak** `discard`, plus penyimpanan `aliases`).

**Yang tidak boleh diambil:** fuzzy/semantic dengan threshold tunggal, closure transitif, dan merge `discard` tanpa jejak.

Temuan ini **memperkuat** rekomendasi di [00-ringkasan-dan-rekomendasi.md](00-ringkasan-dan-rekomendasi.md): skor string atau embedding saja tidak cukup untuk korpus regulasi Indonesia. Keputusan harus dibatasi oleh veto deterministik (digit, discriminator token, tipe) dan dicatat di alias registry. Survivor juga harus dipilih secara eksplisit (mis. nama terpanjang/terlengkap atau ID canonical), bukan diserahkan ke urutan list APOC.

---

## 15. Log revisi (2026-10-17): koreksi setelah verifikasi di Neo4j

| § | Klaim lama | Hasil verifikasi |
|---|---|---|
| 2.1 | Prompt di L162. `name` wajib jika didefinisikan di schema. | Prompt di L177. `name` hanya dipaksa lewat constraint `EXISTENCE`/`KEY`. |
| 4, 8 | Survivor = urutan `collect`, "biasanya urutan pembuatan". | Urutan scan fisik, yang ≠ urutan pembuatan setelah delete (ID reuse). |
| 3, 4, 12 | Label berbeda tidak pernah digabung. Node multi-label hanya "diproses dua kali". | Node multi-label meruntuhkan grup lintas label menjadi satu node. |
| 5 | Survivor ditentukan urutan `set` Python, tidak deterministik antar run Python. | `collect` mengabaikan urutan `$ids`. Deterministik per keadaan fisik DB, `PYTHONHASHSEED` tidak berpengaruh. |
| 5 | — | Baru: `elementId IN $ids` menjadi `AllNodesScan` per set. |
| 5 | Efek bug konsolidasi hanya dinalar. | Terverifikasi di DB: under-merge, bergantung urutan scan. |
| 6 | `ratio` = Levenshtein. Token ratio × 0,95 selalu ikut. | `ratio` = Indel. Token ratio hanya jika rasio panjang < 1,5, dan selain itu `partial_token_ratio` × 0,95 × s. Rasio panjang ICCN 10,0, bukan 9,75. |
| 6, 10.2 | — | Baru: fuzzy menggabungkan "Kementerian Keuangan"↔"Kementerian Perdagangan" (0,84). |
| 7, 10.6 | Semua "Kota X" saling ≥ 0,81 (spaCy). | 0,795–0,904. "Kota Batu"↔"Kota Yogyakarta" < 0,8. |
| 7, 11 | spaCy 16.300 pasangan/detik, ±51 menit untuk 10 rb entity. RSS 805 MB. | ±130.000 pasangan/detik, ±8 menit total. RSS ±740 MB. `nlp()` menjalankan seluruh pipeline. |
| 8 | Properti relasi hasil merge mengikuti `discard` (survivor). | Key digabung, dan nilai yang konflik tidak mengikuti survivor. |
| 9 | — | Baru: contoh `filter_query` `FROM_DOCUMENT` di panduan resmi tidak berpengaruh apa pun. |
| 10.5 | Simulasi Python: kedua scorer 12 → 3 node dengan cluster identik (angka tidak konsisten dengan cluster yang dicantumkan). | Run nyata di Neo4j: spaCy 2 node (stabil), fuzzy 3/3/2 node dengan entity yang lolos berbeda tiap run. |
