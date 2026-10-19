# Analisis Entity Deduplication — `docling-graph`

> Versi package: `docling_graph` 1.9.1 ([pyproject.toml](../../docling-graph/pyproject.toml)). Semua path relatif ke `docling-graph/docling_graph/`.
> - Konfigurasi & pemilihan contract: [config.py](../../docling-graph/docling_graph/config.py), [contracts/auto.py](../../docling-graph/docling_graph/core/extractors/contracts/auto.py), [strategies/many_to_one.py](../../docling-graph/docling_graph/core/extractors/strategies/many_to_one.py)
> - Prompt & gleaning: [direct/prompts.py](../../docling-graph/docling_graph/core/extractors/contracts/direct/prompts.py), [dense/prompts.py](../../docling-graph/docling_graph/core/extractors/contracts/dense/prompts.py), [gleaning.py](../../docling-graph/docling_graph/core/extractors/gleaning.py), [dict_merger.py](../../docling-graph/docling_graph/core/utils/dict_merger.py)
> - Identitas: [entity_name_normalizer.py](../../docling-graph/docling_graph/core/utils/entity_name_normalizer.py), [node_id_registry.py](../../docling-graph/docling_graph/core/converters/node_id_registry.py), [provenance/identity.py](../../docling-graph/docling_graph/core/provenance/identity.py)
> - Dedup dense: [dense/orchestrator.py](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py), [dense/resolvers.py](../../docling-graph/docling_graph/core/extractors/contracts/dense/resolvers.py)
> - Dedup graph: [graph_converter.py](../../docling-graph/docling_graph/core/converters/graph_converter.py), [graph_cleaner.py](../../docling-graph/docling_graph/core/utils/graph_cleaner.py), [alias_reconciler.py](../../docling-graph/docling_graph/core/utils/alias_reconciler.py), [pipeline/stages.py](../../docling-graph/docling_graph/pipeline/stages.py)
> - Fusi multi-graph: [merge/merger.py](../../docling-graph/docling_graph/core/merge/merger.py), [merge/identity.py](../../docling-graph/docling_graph/core/merge/identity.py), [merge/node_folder.py](../../docling-graph/docling_graph/core/merge/node_folder.py), [merge/policy.py](../../docling-graph/docling_graph/core/merge/policy.py)
> - Ekspor: [exporters/cypher_exporter.py](../../docling-graph/docling_graph/core/exporters/cypher_exporter.py)
>
> Semua klaim perilaku dicek dengan membaca kode dan **menjalankan fungsi asli** docling-graph tanpa salinan atau modifikasi (§6).

## Catatan revisi (review ke-2, langsung ke kode)

Review pertama sudah membetulkan cakupan L3 (hanya `dense`), veto sibling, dan granularitas veto co-occurrence. Review kedua membaca ulang seluruh jalur dedup dan menemukan hal berikut:

| # | Klaim versi sebelumnya | Fakta di kode |
|---|---|---|
| 1 | Jalur `direct` = satu panggilan LLM. | `gleaning_enabled=True` (default, [config.py L220](../../docling-graph/docling_graph/config.py#L220)) menambah **panggilan kedua** ("what did you miss?"). Hasilnya digabung dengan `deep_merge_dicts` yang **exact, tanpa normalisasi** (§2.2). |
| 2 | Veto sibling L4 memblokir alias di list yang sama. | Benar, tetapi veto juga jalan bila kedua node punya **child yang sama dengan label edge yang sama**. Alias asli yang berbagi relasi (mis. sama-sama `LOCATED_IN → Bandung`) **ikut diveto**, termasuk di `merge` dengan keputusan manusia (§2.7, §6.3). |
| 3 | Split collision lintas dokumen hanya untuk kasus "baris 2 di dua invoice". | Satu konflik scalar (selain `description`/`summary`) pada **satu** entity membuat **semua** collision kelas yang sama antar pasangan dokumen itu ikut di-split, termasuk entity yang isinya identik. Untuk atribut hasil LLM, dedup lintas dokumen jadi nyaris nol (§2.8, §6.6). |
| 4 | Normalisasi kunci kanonik seragam. | Ada **dua rezim** yang dipilih dari **nama field**: `name`/`title`/`nom` vs field lain (mis. `nama`). Pada rezim kedua, nilai beraksara non-Latin menjadi `""` sehingga semua entity itu **runtuh menjadi satu node** tanpa peringatan (§2.3, §6.1). |
| 5 | Registry hanya menggabungkan entity ber-ID kosong. | Identitas registry = (kelas, field ID). Tidak dibatasi parent atau path. Homonim dalam satu dokumen tergabung dan nilai atribut yang bentrok **dibuang tanpa jejak** (§2.3, §6.4). |
| 6 | GraphCleaner adalah lapisan dedup tersendiri. | Di jalur konversi, lapisan ini **tidak pernah bisa memicu merge**: node ber-ID beda pasti beda isi. Lapisan ini baru bekerja di `docling-graph merge` (§2.6). |
| 7 | `dense_reconciliation.json` selalu ditulis. | Hanya ditulis dengan `--debug` ([stages.py L503](../../docling-graph/docling_graph/pipeline/stages.py#L503)). L4 saat konversi hanya menyimpan **statistik agregat**. Pasangan yang ditolak atau diveto hanya muncul di log (§2.7). |
| 8 | (tidak dibahas) | L2 fuzzy **mengakumulasi** `_source_chunk_ids`, sehingga cakupan veto L3 melebar. L3 dengan > 100 instance dipotong per 100 menurut urutan kemunculan, jadi alias di sub-batch berbeda tidak pernah bertemu (§2.4, §2.5). |
| 9 | (tidak dibahas) | Ekspor Cypher memakai `MERGE (n {id}) SET …`. Memuat beberapa dokumen ke Neo4j lewat jalur ini **melewati semua guard fusi** dan berlaku last-write-wins (§2.9). |
| 10 | (koreksi kecil) | Survivor L4 dipilih dari jumlah atribut terisi. Bila seri, **base (nama pendek/singkatan) yang bertahan**: node akhir bernama "ICCN", bentuk panjang hanya tersimpan di `merged_aliases` (§6.3). Docstring normalizer menyebut dipakai `dict_merger`, padahal tidak. Opsi `identity_fields_map` dan `merge_similarity_fallback` di `dict_merger` tidak dipanggil dari kode produksi mana pun. |

---

## 1. Paradigma dan alur

### 1.1 Identitas dideklarasikan oleh *template*

docling-graph adalah ekstraksi **schema-first**. User menulis template Pydantic, dan setiap kelas entity mendeklarasikan field identitasnya:

```python
class Organization(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])   # bisa lebih dari satu field
    name: str
```

Node ID = `f"{Class}_{blake2b(json({field: canonical(value)…, '__class__': Class}))[:16]}"` ([`NodeIDRegistry._generate_fingerprint` L58](../../docling-graph/docling_graph/core/converters/node_id_registry.py#L58)). Kelas tanpa `graph_id_fields` di-fingerprint dari semua field **scalar yang truthy**, sehingga list, dict, `0`, dan `False` tidak ikut dihitung.

Prinsipnya **memisahkan identity key dari display name**: atribut node menyimpan nilai asli, sedangkan ID memakai bentuk kanonik.

### 1.2 Contract ekstraksi menentukan lapisan dedup yang aktif

Default `extraction_contract="auto"` ([config.py L102](../../docling-graph/docling_graph/config.py#L102)) diputuskan per dokumen oleh [`resolve_auto_contract` L66](../../docling-graph/docling_graph/core/extractors/contracts/auto.py#L66):
- **`direct`** bila estimasi input (`chars // 4`) ditambah budget output muat di context window **dan** `chars ≤ output_budget × 4`. Contoh: budget output 8k token berarti dokumen ≤ ~32k karakter. Dokumen diekstrak dalam satu panggilan, lalu **gleaning** satu kali (default aktif).
- **`dense`** selain itu: skeleton per batch chunk (Phase 1), lalu fill (Phase 2).

```
                ┌──────────── direct (dokumen kecil) ──────────────────┐
Dokumen ─auto─► │ L0 prompt rule 8 → 1 respons utuh                    │
                │ L0g gleaning → deep_merge_dicts (exact, tanpa norm.)  │
                └──────────────────────────────────────────────────────┘
                ┌──────────── dense (dokumen besar) ───────────────────┐
                │ L0 already_found (hanya sequential)                  │
                │ L1 kunci (path, id kanonik) antar batch              │
                │ L2 fuzzy OCR (hanya aggressive)                      │
                │ L3 rekonsiliasi skeleton LLM (standard+, default)    │
                └──────────────────────────────────────────────────────┘
                                   │
                                   ▼  (kedua contract)
     Graph conversion: L1 registry (kelas + id kanonik, first-non-empty, konflik dibuang)
                     → GraphCleaner (hash konten exact; di jalur ini tak pernah aktif)
                     → L4 alias (containment → LLM → veto sibling/shared-child)
                                   │
                   ┌───────────────┴────────────────┐
                   ▼ (opsional, CLI)                 ▼ (opsional)
   `docling-graph merge`: re-key → skolem root    Ekspor Cypher: MERGE on id
   → split konflik (menular per kelas)            + SET (last-write-wins,
   → union fold → cleaner → L4 keputusan manusia   tanpa guard)
```

Backend VLM menggabungkan model per halaman dengan `merge_pydantic_models` ([many_to_one.py L173](../../docling-graph/docling_graph/core/extractors/strategies/many_to_one.py#L173)), yang memakai `deep_merge_dicts` yang sama dengan gleaning. Backend VLM tidak punya `call_id_reconciliation`, sehingga L4-nya propose-only.

---

## 2. Lapisan dedup, satu per satu

### 2.1 L0: pencegahan di prompt ekstraksi

- **direct**: rule 8 *"Keep identifiers consistent across references (same entity => same identifier value)"* ([direct/prompts.py L48](../../docling-graph/docling_graph/core/extractors/contracts/direct/prompts.py#L48)). LLM melihat seluruh dokumen sekaligus, jadi inilah mekanisme utama penyatuan varian nama di jalur direct. Sifatnya probabilistik dan tidak bisa diaudit.
- **dense**: rule 3 meminta ID berupa *"short labels copied verbatim from the document"* ([dense/prompts.py L88](../../docling-graph/docling_graph/core/extractors/contracts/dense/prompts.py#L88)). Varian penamaan antar batch justru **diharapkan** dan diserahkan ke L1–L3.
- **dense sequential**: prompt menampilkan ≤ 50 entity terakhir (`_ALREADY_FOUND_WINDOW`, [L72](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L72)) dengan instruksi "DO NOT RE-OUTPUT". Mekanisme ini **mati pada default** `parallel_workers=4` ([L2470](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L2470)). Catatan: nilai `0` dibaca sebagai `4` (`int(... or 4)`), jadi mode sequential harus diset `1`.

### 2.2 L0g: gleaning dan penggabungan dict (direct dan VLM)

[`run_gleaning_pass_direct`](../../docling-graph/docling_graph/core/extractors/gleaning.py#L63) mengirim ulang dokumen beserta hasil pertama (dipotong di **8000 karakter**) dan meminta entity tambahan. Hasilnya digabung dengan [`deep_merge_dicts`](../../docling-graph/docling_graph/core/utils/dict_merger.py#L76):
- Item list entity dicocokkan dengan **hash JSON seluruh isi** (`identity_fields_map` tidak pernah dioper oleh pemanggil produksi mana pun). Entity yang sama dengan deskripsi berbeda, atau `"ICCN"` vs `"iccn"`, menjadi item terpisah. Item ini baru disatukan registry (L1) jika field ID-nya sama secara kanonik (§6.5).
- Scalar non-deskripsi **ditimpa** oleh hasil gleaning (`target[key] = source_value`). Contohnya judul root "Laporan A" berubah menjadi "Laporan A (rev)". `description`/`summary` digabung per kalimat.
- Bila hasil pertama > 8000 karakter, LLM gleaning tidak melihat semua entity yang sudah ada, sehingga peluang muncul varian nama baru meningkat.

Jadi lapisan ini **tidak menyumbang dedup varian**. Fungsinya menambah recall, dan dedup tetap bergantung pada L1/L4.

### 2.3 L1: kunci kanonik deterministik — [`canonicalize_identity_for_dedup` L58](../../docling-graph/docling_graph/core/utils/entity_name_normalizer.py#L58)

Rezim normalisasi dipilih dari **nama field**, bukan tipe datanya:

| Rezim | Field | Langkah | Contoh |
|---|---|---|---|
| Nama | `name`, `title`, `nom` | NFKD → buang diakritik → buang artikel awal `The/A/An` → split spasi/`_`/`-` → buang `'s` → `UPPER_SNAKE` | `"Bekraf's"`→`BEKRAF`, `"UU No. 24"`→`UU_NO._24` |
| Lain | semua field lain (`nama`, `kode`, …) | NFKD → casefold → **hanya `[a-z0-9]`** | `"Kota Malang"`=`"KotaMalang"`→`kotamalang`, `"Bekraf's"`→`bekrafs` |

Akibat yang perlu dicatat (terverifikasi, §6.1):
- Template berbahasa Indonesia dengan field `nama` **tidak** mendapat pembuangan artikel atau posesif, tetapi lebih toleran terhadap spasi dan tanda baca.
- Pada rezim "lain", nilai yang seluruhnya **non-Latin** (`"東京"`, `"大阪"`, aksara Jawa) menjadi `""`. Semuanya mendapat node ID yang sama dan **tergabung menjadi satu node**. Peringatan `empty_identity_nodes` tidak muncul karena nilai mentahnya tidak kosong.
- Rezim nama mempertahankan tanda baca selain `_`/`-`: `"UU No. 24"` ≠ `"UU No 24"`, dan `"(ICCN)"` tetap utuh.
- Tidak ada pengenalan bentuk badan usaha Indonesia (`PT`, `Tbk`, `(Persero)`).

Fungsi kanonik yang sama dipakai di skeleton dedup ([`identity_pairs`](../../docling-graph/docling_graph/core/provenance/identity.py#L29)), registry, proposer containment (L3/L4), dan re-key fusi ([`recompute_node_id`](../../docling-graph/docling_graph/core/merge/identity.py#L86), dijaga parity test). Konsistensi lintas tahap ini kekuatan nyata.

Ada **dua kunci identitas** yang cakupannya berbeda:
- **Skeleton dense**: `(path katalog, id kanonik)` ([`_skeleton_identity_key` L210](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L210)). Entity yang sama di dua path baru menyatu di registry.
- **Registry graph**: `(kelas, id kanonik)`, **tanpa parent dan tanpa path**. Dua `Organization(name="Dinas Pariwisata")` di bawah kota yang berbeda menjadi satu node. Penggabungan memakai [`_enrich_existing_node` L656](../../docling-graph/docling_graph/core/converters/graph_converter.py#L656) (**first non-empty wins**): atribut kosong diisi, sedangkan atribut yang bentrok **dibuang tanpa catatan** (kota "Batu" hilang, §6.4). Berbeda dengan `merge`, yang mencatat setiap konflik.

ID kosong:
- Registry mengubah `None`/`""` menjadi `""`, sehingga semua instance satu kelas yang field ID-nya kosong menjadi **satu node**. Peringatan `empty_identity_nodes` ([L329](../../docling-graph/docling_graph/core/converters/graph_converter.py#L329)) muncul **setelah** penggabungan.
- Skeleton dense: `ids={"name": ""}` juga menghasilkan kunci yang sama (`("name","")`). Fallback per-node hanya berlaku bila *tidak ada* pasangan sama sekali.

### 2.4 L2: fuzzy untuk OCR noise — [`resolve_skeleton_nodes` L144](../../docling-graph/docling_graph/core/extractors/contracts/dense/resolvers.py#L144)

- Hanya aktif pada `dense` + `dense_dedupe="aggressive"` (default `standard`, [config.py L244](../../docling-graph/docling_graph/config.py#L244)).
- Pasangan dalam path yang sama (O(n²)), melewati pasangan yang kunci kanoniknya sudah sama.
- **Digit-signature guard**, lalu `rapidfuzz.token_sort_ratio ≥ 0.90` pada nilai ID **mentah** yang digabung `" | "`. Nilai yang digabung adalah semua `ids`, tidak hanya `id_fields`. Threshold sengaja tidak bisa diatur. rapidfuzz adalah dependensi wajib; fallback `SequenceMatcher` hanya jaga-jaga.
- Skor case-sensitive ("Kota Malang" vs "kota malang" = 81,8), tetapi kapitalisasi sudah disatukan di L1.
- Greedy: node yang lebih awal menyerap node berikutnya, dan teks pembandingnya tidak berubah. Parent di-remap penuh.
- **`_absorb_sources` mengunion `_source_chunk_ids`** ([L81](../../docling-graph/docling_graph/core/extractors/contracts/dense/resolvers.py#L81)), berbeda dengan L1 yang hanya menyimpan chunk emisi pertama. Node hasil L2 membawa chunk lebih banyak, sehingga lebih mudah terkena veto co-occurrence L3 (§6.7).

### 2.5 L3: rekonsiliasi skeleton via LLM — [`_run_skeleton_reconciliation` L1742](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L1742)

Aktif pada `dense` dengan `dense_dedupe` ≠ `off` (default `standard`).

1. Instance dikelompokkan per path (root dikecualikan; path dengan < 2 instance dilewati).
2. **Batching** ([L99](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L99)): satu panggilan bila total ≤ 100 instance. Jika lebih, panggilan dibuat per path dan dipotong per 100 menurut **urutan kemunculan**. Alias yang jatuh di sub-batch berbeda **tidak pernah dibandingkan**. Respons kosong pada list ≥ 100 dicatat sebagai warning.
3. Proposer containment per sub-batch disertakan sebagai **saran** di prompt.
4. LLM hanya menerima **daftar identifier bernomor** (tanpa isi dokumen) dan mengembalikan `{"path","keep","merge":[...]}`, dengan `keep` = instance **paling spesifik** ([prompt](../../docling-graph/docling_graph/core/extractors/contracts/dense/prompts.py#L136)). Prompt melarang merge yang berbeda parameter/angka/versi dan pasangan tier.
5. **Tidak ada guard proposer ∩ LLM**: LLM bebas menggabungkan pasangan apa pun dalam satu path. Pembatas kerasnya hanya veto di langkah 6.
6. **Veto co-occurrence** ([`apply_skeleton_reconciliation` L496](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L496)): merge ditolak bila `_source_chunk_ids` kedua instance beririsan.
   - `_source_chunk_ids` = chunk dari **batch pertama** yang memunculkan node ([L367](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L367)), atau satu chunk bila LLM memberi atribusi `"c"` yang valid untuk batch itu ([L317](../../docling-graph/docling_graph/core/extractors/contracts/dense/orchestrator.py#L317)), atau gabungan chunk setelah L2.
   - Akibatnya, dua alias yang pertama kali muncul di batch yang sama (±2048 token) tanpa atribusi `"c"` **pasti diveto**, termasuk pola definisi "Indonesia Creative Cities Network (ICCN)" yang diikuti "ICCN" (§6.7).
7. Grup yang `keep`-nya sudah diserap grup sebelumnya dilewati utuh, jadi tidak ada rantai merge.
8. Lineage: `_reconciled_chunk_ids`, `_merged_from`, dan parent di-remap. File `dense_reconciliation.json` **hanya ditulis bila `--debug`**.
9. Respons invalid diabaikan: pass ini hanya bisa merge, tidak pernah menghapus node.

### 2.6 GraphCleaner — [`_deduplicate_nodes` L273](../../docling-graph/docling_graph/core/utils/graph_cleaner.py#L273)

Berjalan bila `auto_cleanup=True` (default). Node dikelompokkan berdasarkan hash JSON atribut non-`None` di luar metadata (`id`, `label`, `type`, provenance, `merged_*`, `__conflicts__`; `__class__` ikut dihitung). Node identik digabung ke node pertama, edge dialihkan, dan provenance di-union. Placeholder `"Unknown"` mencampurkan node ID ke hash ([L387](../../docling-graph/docling_graph/core/utils/graph_cleaner.py#L387)).

**Di jalur konversi, merge ini tidak pernah bisa terjadi.** Node ID adalah hash dari subset isi yang sama: field ID mentah (atau scalar truthy untuk kelas tanpa ID) ikut di dalam hash konten. Isi yang identik berarti ID identik, dan node dengan ID identik sudah menjadi satu node di registry. Lapisan ini baru berfungsi di `docling-graph merge`, tempat node di-split atau di-skolemize dengan ID baru. Karena itu split diberi cap `skolem_document_id` ([merger.py L437](../../docling-graph/docling_graph/core/merge/merger.py#L437)) agar tidak digabung lagi oleh cleaner.

### 2.7 L4: rekonsiliasi alias level graph — [`reconcile_graph_aliases` L295](../../docling-graph/docling_graph/core/utils/alias_reconciler.py#L295)

Dipanggil setelah cleanup ([graph_converter.py L303](../../docling-graph/docling_graph/core/converters/graph_converter.py#L303)) untuk **kedua contract**. Pipeline otomatis memasang `alias_llm_fn = backend.call_id_reconciliation` ([stages.py L797](../../docling-graph/docling_graph/pipeline/stages.py#L797)). Backend tanpa method itu (VLM) menjalankan mode **propose-only**.

1. **Propose** ([`containment_groups` L59](../../docling-graph/docling_graph/core/utils/alias_reconciler.py#L59)), per kelas, atas teks kanonik (beberapa field ID digabung **tanpa pemisah**). Kandidat = A **substring karakter** dari B, dengan syarat:
   - digit signature sama;
   - base ≥ 4 karakter dan lebih pendek dari superset;
   - base unik per superset. Superset dengan dua base dilewati, sehingga pada rantai "Malang" ⊂ "Kota Malang" ⊂ "Pemerintah Kota Malang" hanya pasangan pertama yang diusulkan.
   - Satu base boleh punya banyak superset ("Malang" → "Kota Malang" **dan** "Kabupaten Malang").
2. **Confirm**: **satu panggilan LLM untuk seluruh graph, tanpa batching** ([prompt L159](../../docling-graph/docling_graph/core/utils/alias_reconciler.py#L159)). Prompt memuat semua identifier dari setiap kelas yang punya kandidat, tanpa deteksi "silent 0 merges" seperti di L3. Kegagalan panggilan berarti tidak ada merge.
3. **Guard rail**: hanya pasangan usulan proposer yang boleh di-merge (proposer ∩ LLM).
4. **Veto sibling/shared-neighbor** ([`_sibling_co_occurrence` L238](../../docling-graph/docling_graph/core/utils/alias_reconciler.py#L238)): veto bila ada **parent bersama** yang menunjuk keduanya dengan label sama, **atau child bersama** yang ditunjuk keduanya dengan label sama.
   - Anggota list yang sama (`organizations: List[...]`) **selalu** diveto.
   - Alias asli yang berbagi relasi keluar (keduanya `LOCATED_IN → Bandung`, keduanya `PART_OF → X`) **juga diveto**. Padahal relasi yang sama justru bukti pendukung alias. Veto berlaku juga untuk keputusan manusia di `merge` (§6.3, §6.6).
5. Survivor = node dengan atribut terisi terbanyak; **bila seri, base (nama pendek) menang**. Atribut kosong diisi, identitas yang diserap dicatat di `merged_aliases`, dan edge dialihkan.
6. Tidak ada closure transitif otomatis. Node yang sudah diserap dilewati, tetapi survivor tidak dikunci.
7. Jejak audit saat konversi hanya `graph.graph["alias_reconciliation"]` (hitungan `candidates/confirmed/merged/vetoed_sibling`) dan `merged_aliases` di node. Daftar pasangan yang ditolak LLM atau diveto **hanya ada di log**.

### 2.8 L5: fusi multi-graph (`docling-graph merge`) — [`GraphMerger.merge` L242](../../docling-graph/docling_graph/core/merge/merger.py#L242)

Urutan: re-key → skolemize root → split konflik → union fold → GraphCleaner → alias pass.

1. **Re-key** ([`rekey_graph` L151](../../docling-graph/docling_graph/core/merge/identity.py#L151)): ID dihitung ulang dengan normalizer versi sekarang. Fan-in di-fold, dan collision lintas kelas menjadi error.
2. **Skolemize root** ([L358](../../docling-graph/docling_graph/core/merge/merger.py#L358)): root dari dua dokumen berbeda yang ID-nya sama karena berasal dari nama file diberi ID baru. Tanpa ledger, root tetap digabung dengan warning.
3. **Split collision berbasis konflik** ([`_split_conflicting_collisions` L473](../../docling-graph/docling_graph/core/merge/merger.py#L473)). Fold node ber-ID sama dibatalkan bila (a) kedua kemunculan tidak berbagi root dan (b) ada konflik scalar ([`conflicting_scalar_fields`](../../docling-graph/docling_graph/core/merge/node_folder.py#L113)). Yang dikecualikan dari konflik: `description`/`summary` (digabung per kalimat), list (di-union), dan beda kapitalisasi/spasi/bentuk Unicode.
   - **Contagion**: satu konflik pada kelas C antara dokumen A dan B membuat **semua** collision kelas C antara A dan B di-split, termasuk yang isinya identik. Contoh: `ICCN.role` "penyelenggara" vs "anggota" juga memecah `Bekraf` yang identik di kedua dokumen (§6.6).
   - Pada atribut hasil LLM (`role`, `type`, `sector`, …) perbedaan kecil antar dokumen lazim. Akibatnya dedup lintas dokumen untuk kelas tersebut **praktis berhenti**. Ini precision-first yang ekstrem.
   - Node hasil split bernama `<id>__doc_<id8>` (atau `__src_<i>` tanpa ledger) dan diberi cap `skolem_document_id`.
4. **Union fold** ([`_union_fold` L601](../../docling-graph/docling_graph/core/merge/merger.py#L601)): node ber-ID sama digabung dengan policy `precedence` (`input-order`/`richest`) dan `conflicts` (`keep-first`/`keep-all`/`variants`). Opsi terakhir mereifikasi nilai konflik sebagai sub-node `HAS_CONFLICT_VARIANT`. Dedup lintas dokumen di sini **exact pada kunci kanonik**.
5. **Alias pass tanpa LLM** ([`_alias_pass` L720](../../docling-graph/docling_graph/core/merge/merger.py#L720)): kandidat containment ditulis ke `merge_report.json` sebagai stub (`confirm:false`, skor `token_sort_ratio` sebagai triase, `field_conflicts`). User mengubahnya menjadi `confirm:true` lalu menjalankan ulang dengan `--alias-decisions` ([`_build_decisions_fn` L811](../../docling-graph/docling_graph/core/merge/merger.py#L811)).
   - Keputusan dikunci pada node ID deterministik dan bisa di-*replay*, **asal pasangan itu diusulkan lagi** oleh proposer pada run tersebut.
   - Pasangan yang tidak diusulkan atau diveto guard dilaporkan di `ignored_alias_decisions`.
   - `merged_aliases` **tidak pernah dipakai sebagai kunci identitas** di merge berikutnya. Bentuk panjang di dokumen baru tetap node terpisah sampai dikonfirmasi lagi.
   - CLI `merge` tidak punya opsi LLM. Satu-satunya confirmer adalah manusia.

### 2.9 Ekspor Cypher (jalur alternatif ke Neo4j)

[`CypherExporter`](../../docling-graph/docling_graph/core/exporters/cypher_exporter.py#L277) default `style="merge"`: `MERGE (n:Label {id: …}) SET n.prop = …`. Memuat ekspor beberapa dokumen langsung ke satu database menggabungkan node ber-ID sama **tanpa** split konflik, skolemisasi, atau catatan konflik. Atribut menjadi **last-write-wins**. Guard di §2.8 hanya berlaku bila graph digabung dulu dengan `docling-graph merge`.

---

## 3. Kekuatan (layak diadopsi)

1. **Identity key ≠ display name** dengan satu fungsi kanonik yang dipakai konsisten di semua tahap (skeleton, registry, proposer, re-key) dan dijaga parity test.
2. **Filosofi precision-first** yang ditegakkan di kode, bukan hanya di prompt: *"A false merge destroys data; a kept duplicate is merely redundant."*
3. **LLM sebagai verifier** di L4 (proposer ∩ LLM). Di L3, LLM lebih bebas tetapi tetap dibatasi veto deterministik.
4. **Guard deterministik yang tidak bisa dibatalkan LLM** (bahkan oleh manusia): digit signature, base unik, co-occurrence chunk/batch, sibling dengan label sama, split konflik per kelas.
5. **LLM id-space** (tanpa isi dokumen). L3 di-batch per ≤ 100 identifier, dengan peringatan bila respons kosong pada list besar.
6. **Human-in-the-loop yang bisa di-replay** dengan node ID deterministik, plus laporan keputusan yang diabaikan beserta alasannya.
7. **Provenance merge di tahap fusi**: `merged_from`, `__conflicts__`, varian konflik sebagai sub-node, laporan split dan konflik.

## 4. Kelemahan (untuk kebutuhan hybrid_rag)

### 4.1 [TINGGI] Recall bergantung pada contract dan mode, dan default-nya sempit
- **Dokumen kecil (direct)**: penyatuan varian sepenuhnya bergantung pada LLM ekstraksi (rule 8). Gleaning menambah entity tanpa normalisasi. Setelah itu tersisa L1 (exact kanonik) dan L4 (containment).
- **Dokumen besar (dense)**: L3 ada, tetapi token-swap deterministik (L2) hanya pada `aggressive`, dan L3 di atas 100 instance hanya membandingkan dalam potongan urutan kemunculan.
- Pada L4 dan fusi, **satu-satunya proposer adalah substring containment**. Tidak ada token-bag, akronim/inisial, atau embedding. Pasangan non-substring tidak bisa di-merge, bahkan oleh keputusan manusia.

### 4.2 [TINGGI] Veto memblokir alias yang sah
- **Veto L4 sibling/shared-neighbor**: anggota list yang sama **selalu** diveto. Alias asli yang berbagi child dengan label edge sama juga diveto (terverifikasi di konversi dan di `merge` dengan keputusan manusia, §6.3, §6.6). Pada graph yang kaya relasi, dua node yang menunjuk entitas yang sama dengan relasi yang sama adalah pola umum alias.
- **Veto co-occurrence L3**: granularitasnya per batch bila tidak ada atribusi chunk, dan melebar setelah L2. Definisi singkatan beserta pemakaian pertamanya hampir pasti diveto.

### 4.3 [TINGGI] Dedup lintas dokumen nyaris mati oleh contagion
Satu konflik scalar pada satu entity memecah semua collision kelas tersebut antar pasangan dokumen (§2.8, §6.6). Untuk entity dengan atribut hasil LLM, hasil `merge` cenderung berisi duplikat per dokumen. Di sisi lain, jalur Cypher `MERGE` melompati semua guard (last-write-wins, §2.9). Pilihannya hanya terlalu ketat atau tanpa guard sama sekali.

### 4.4 [TINGGI] Tidak ada alias registry buatan user
Tidak ada cara mendeklarasikan "ICCN ≡ Indonesia Creative Cities Network" atau "Kemenparekraf ≡ Kementerian Pariwisata dan Ekonomi Kreatif" sebagai aturan saat ekstraksi. Decision file hanya berlaku di `merge`, hanya untuk pasangan containment, dan `merged_aliases` tidak pernah dipakai ulang sebagai identitas.

### 4.5 [SEDANG] Penggabungan diam-diam di registry
- **Homonim** dalam satu dokumen: identitas (kelas, id) tidak dibatasi parent. Nilai atribut yang bentrok dibuang tanpa catatan (first non-empty wins, §6.4).
- **ID kosong** (`None`/`""`) digabung menjadi satu node. Peringatan baru muncul setelah penggabungan.
- **Nilai non-Latin** pada field ID yang bukan `name`/`title`/`nom` dikanonikkan menjadi `""` dan digabung menjadi satu node **tanpa peringatan** (§6.1). Risikonya tergantung template: kecil untuk teks Indonesia berhuruf Latin, tetapi nyata untuk nama Arab/aksara daerah atau kode yang seluruhnya simbol.
- Rezim normalisasi bergantung pada **nama field**. Template dengan `nama` mendapat perilaku berbeda dari template dengan `name`.

### 4.6 [SEDANG] Lingkup per dokumen, tanpa blocking semantik
- L2–L4 bekerja di dalam satu dokumen. Sub-batching L3 berdasarkan urutan kemunculan, bukan kemiripan.
- L4 tanpa batching: pada graph besar satu prompt bisa sangat panjang, dan ini mode kegagalan yang diwaspadai di L3.

### 4.7 [SEDANG] Proposer containment per karakter
Tidak ada pemeriksaan batas kata: "Seni" ⊂ "Kesenian Daerah", "Malang" ⊂ "Kabupaten Malang", dan "ICCN" ⊂ "ICCN dan JKN" ("JKN" < 4 karakter sehingga tidak dihitung ambigu). Semuanya menjadi kandidat. Keamanannya bergantung pada LLM dan veto. Batasan per kelas mencegah Location vs Organization saling diusulkan, **asalkan** tipenya benar.

### 4.8 [SEDANG] LLM hanya melihat identifier, audit L4 tipis
- Nama saja tidak cukup untuk homonim ("Malang" kota vs kabupaten). recon_graphrag menyertakan deskripsi di tahap ini.
- Saat konversi, keputusan L4 yang ditolak atau diveto tidak dipersist, dan forensik L3 hanya tersedia dengan `--debug`.

### 4.9 [RENDAH] Ketergantungan template dan paradigma
Seluruh mekanisme membutuhkan `graph_id_fields` pada template Pydantic dan graph networkx in-memory per dokumen. hybrid_rag memakai ekstraksi open-schema per chunk dengan Neo4j yang inkremental. Yang bisa dipindahkan adalah **pola**, bukan kodenya.

---

## 5. Peta lapisan vs kondisi aktif

| Lapisan | direct (default dok. kecil) | dense (default dok. besar) | Butuh LLM | Deterministik | Jejak audit |
|---|---|---|---|---|---|
| L0 prompt (konsistensi ID) | ✓ rule 8 | rule 3 (verbatim) | (bagian ekstraksi) | Tidak | – |
| L0 already_found | – | Hanya `parallel_workers=1` | (bagian ekstraksi) | Tidak | – |
| L0g gleaning merge | ✓ (default on) | – | Ya (ekstraksi ke-2) | Exact hash | – |
| L1 kunci kanonik skeleton | – | ✓ | Tidak | Ya | `_source_batch_indexes` |
| L1 registry | ✓ | ✓ | Tidak | Ya | Tidak ada (konflik dibuang) |
| L2 fuzzy OCR | – | Hanya `aggressive` | Tidak | Ya | `_merged_from` |
| L3 rekonsiliasi skeleton | – | ✓ (`standard`) | Ya | Tidak (dibatasi veto) | `_merged_from`; JSON hanya `--debug` |
| GraphCleaner | (tak pernah memicu) | (tak pernah memicu) | Tidak | Ya | – |
| L4 alias graph | ✓ (LLM backend) | ✓ (LLM backend) | Ya (propose-only tanpa LLM) | Tidak (proposer + veto) | Statistik + `merged_aliases` |
| L5 fusi | CLI `merge` | CLI `merge` | Tidak (manusia) | Ya | Lengkap (`merge_report.json`) |
| Ekspor Cypher MERGE | Opsional | Opsional | Tidak | Ya (last-write-wins) | – |

---

## 6. Uji empiris (fungsi asli docling-graph)

Modul diimpor dari `docling-graph/`. Dependensi berat yang tidak dipakai oleh jalur yang diuji (`docling_core`, `docling`, `litellm`, …) diganti stub di level import. Fungsi yang diuji tidak disalin dan tidak diubah. Skrip: `verify.py` dan `verify2.py` di scratchpad sesi. Untuk L4, dipakai LLM tiruan yang **menyetujui setiap kandidat**, agar efek guard deterministik terlihat terpisah.

### 6.1 Normalisasi: rezim `name` vs field lain

| A | B | `name` sama? | `nama` sama? |
|---|---|---|---|
| Bekraf | Bekraf's | **Ya** (`BEKRAF`) | Tidak (`bekraf` vs `bekrafs`) |
| The Jakarta Post | Jakarta Post | **Ya** | Tidak (`thejakartapost`) |
| Kota Malang | KotaMalang | Tidak | **Ya** (`kotamalang`) |
| Kota Malang | kota-malang | Ya | Ya |
| UU No. 24 | UU No 24 | Tidak (`UU_NO._24`) | **Ya** (`uuno24`) |
| PT Gojek Tbk | Gojek | Tidak | Tidak |

Registry dengan `graph_id_fields=["nama"]`: `東京`, `大阪`, dan `ꦪꦺꦴꦒꦾ` (aksara Jawa) **semuanya** mendapat `Kota_40bffe2d38c80634`, jadi satu node.

### 6.2 L2 `resolve_skeleton_nodes` (aggressive) dan skor fuzzy

| A | B | Kunci L1 sama? | token_sort | Merge L2? | Seharusnya |
|---|---|---|---|---|---|
| Program KaTa Kreatif | Program Kata Kreatif | **Ya** | 95 | (sudah satu di L1) | merge ✓ |
| Kota Malang | Malang Kota | Tidak | 100 | **Ya** | merge ✓ |
| ICCN | Indonesia Creative Cities Networks (ICCN) | Tidak | 17,8 | Tidak | merge ✗ |
| Kota Malang | Kabupaten Malang | Tidak | 66,7 | Tidak | tidak ✓ |
| Kota Malang | Pemerintah Kota Malang | Tidak | 66,7 | Tidak | tidak ✓ |
| Jawa Timur | Jawa Barat | Tidak | 40 | Tidak | tidak ✓ |
| Pasal 9 | Pasal 10 | Tidak | 80 | Tidak (digit guard) | tidak ✓ |
| Bank Indonesia | Bank Indonesia Tbk | Tidak | 87,5 | Tidak | ambigu |
| Kementerian Keuangan | Kementerian Perdagangan | Tidak | 83,7 | Tidak | tidak ✓ |
| Dinas Pariwisata Kota Malang | Dinas Pariwisata Kota Batu | Tidak | 77,8 | Tidak | tidak ✓ |

### 6.3 Proposer `containment_groups` dan L4 `reconcile_graph_aliases` end-to-end

Proposer (kunci kanonik `name`):

| Himpunan node | Kandidat yang diusulkan |
|---|---|
| ICCN, Indonesia Creative Cities Network (ICCN) | ICCN → (ICCN) ✓ |
| ICCN, Indonesia Creative Cities Network | – (miss) |
| Malang, Kota Malang, Pemerintah Kota Malang | Malang → Kota Malang saja (Pemkot ambigu, dilewati) |
| Kota Malang, Pemerintah Kota Malang | Kota Malang → Pemkot (false candidate jika satu kelas) |
| Seni, Kesenian Daerah | Seni → Kesenian Daerah (substring dalam kata) |
| ICCN, ICCN dan JKN | ICCN → nama gabungan (false candidate) |
| ICCN, Bekraf, ICCN dan Bekraf | – (dua base, ambigu) |

L4 untuk ICCN ↔ "Indonesia Creative Cities Network (ICCN)" melalui `GraphConverter.pydantic_list_to_graph`:

| Topologi | Statistik | Node akhir |
|---|---|---|
| a) Keduanya di `organizations` (list sama) | merged=0, **vetoed_sibling=1** | 2 node |
| b) `organizations` vs `partners` (label beda) | merged=1 | "ICCN" (+alias panjang) |
| c) Seperti (b), tetapi keduanya `LOCATED_IN → Bandung` | merged=0, **vetoed_sibling=1** | 2 node |
| d) Seperti (b), atribut sama banyak | merged=1 | **"ICCN"** bertahan (base menang saat seri) |
| e) Seperti (b), bentuk panjang punya `city` | merged=1 | "Indonesia … (ICCN)" bertahan |
| Tanpa LLM | candidates=1, merged=0 | propose-only |

### 6.4 Registry: ID kosong dan homonim

| Input | Hasil |
|---|---|
| `Organization(name=None, city="Malang")`, `(name=None, city="Surabaya")`, `(name="", city="Batu")` | **1 node** (`city="Malang"`), ditandai `empty_identity_nodes` |
| `"Dinas Pariwisata"` (city Malang, list `organizations`) + `"dinas  pariwisata"` (city Batu, list `partners`) | **1 node**, `city="Malang"`; "Batu" hilang tanpa catatan konflik |
| Skeleton dense `ids={"name": ""}` × 2 | kunci identik `('organizations[]', (('name',''),))` |

### 6.5 Gleaning merge (`deep_merge_dicts`)

Hasil pertama `{"title":"Laporan A","organizations":[{ICCN,"jaringan kota"}]}` digabung dengan hasil gleaning `{"title":"Laporan A (rev)","organizations":[{ICCN,"jaringan kota kreatif"},{iccn,"jaringan kota"}]}`. Hasilnya `title="Laporan A (rev)"` (ditimpa) dan list berisi **3 item** (`ICCN`, `ICCN`, `iccn`). Ketiganya baru disatukan oleh registry karena kunci kanoniknya sama.

### 6.6 Fusi `docling-graph merge`

| Skenario | Hasil |
|---|---|
| Dok A: ICCN(role=penyelenggara), Bekraf(role=sponsor). Dok B: ICCN(role=anggota), Bekraf(role=sponsor) | **4 node Organization**, folded=0. ICCN split `field-conflict`, Bekraf split `same-class-conflict` (contagion) |
| Seperti di atas, tetapi di Dok B hanya `description` ICCN yang berbeda | 2 node (fold normal, deskripsi digabung) |
| Dok A: ICCN, Dok C: "Indonesia … (ICCN)" | 1 stub kandidat (similarity 0,18). Dengan `--alias-decisions` confirm → merged=1 |
| Seperti di atas, tetapi keduanya `LOCATED_IN → Bandung` | Keputusan manusia **diveto** (`vetoed_sibling=1`, masuk `ignored_alias_decisions`) |

### 6.7 Skeleton dense: L1 → L2 → veto L3

Batch 0 (chunk 0–2, tanpa atribusi `"c"`) berisi ICCN dan "Indonesia … (ICCN)". Batch 1 (chunk 3–4) berisi "Kota Malang" dan "iccn". Batch 2 (chunk 5) berisi "Malang Kota".
- L1: "iccn" menyatu ke ICCN, dengan chunk tetap `[0,1,2]` (emisi pertama).
- L2: "Malang Kota" menyatu ke "Kota Malang", chunk menjadi `[3,4,5]` (**diakumulasi**).
- L3: LLM meminta ICCN digabung ke bentuk panjang, hasilnya **`vetoed_cooccurrence`** (berbagi chunk 0–2).

### 6.8 Ringkasan untuk kasus target

| Kasus | Ditangani? | Syarat |
|---|---|---|
| ICCN ↔ "… (ICCN)" | Kadang | L4: hanya jika bukan anggota list yang sama **dan** tidak berbagi child berlabel sama. L3: hanya jika muncul pertama di batch berbeda (atau ada atribusi chunk). |
| ICCN ↔ "… Network" (tanpa "(ICCN)") | Jarang | Hanya L3 (LLM bebas) atau rule 8 di direct. |
| Kota Malang ↔ Malang Kota | Kadang | Deterministik hanya dense + aggressive. Dengan field `nama`, hanya "KotaMalang"/"Kota-Malang" yang tertangani, bukan urutan terbalik. |
| Kemenparekraf ↔ bentuk panjang | Jarang | Hanya LLM (L3/rule 8). |
| Entity sama di dua dokumen | Rapuh | Fold exact. Satu konflik atribut pada kelasnya memecah semua. |
| Negatif (Kabupaten, Pasal, Jawa Barat) | Aman di lapisan alias | Digit guard, proposer sempit, veto. Homonim bernama persis sama tetap digabung oleh registry. |

---

## 7. Penilaian

| Dimensi | Skor (1–5) | Catatan |
|---|---|---|
| Recall kasus target | 2 | Deterministik hanya containment dan token-swap (aggressive). Veto sibling/shared-child dan co-occurrence memblokir alias sah. Dedup lintas dokumen rapuh karena contagion. |
| Precision / keamanan | 3,5 | Lapisan alias sangat aman (proposer ∩ LLM, digit guard, veto, split konflik). Lapisan dasar justru menggabungkan diam-diam: homonim lintas parent, ID kosong, nilai non-Latin, dan jalur Cypher last-write-wins. |
| Kustomisasi | 2 | Mode dedupe dan decision file (containment-only, saat merge). Tidak ada alias dict, never-merge, atau pengaturan field mana yang boleh berkonflik. |
| Efisiensi | 4 | LLM id-space. L3 di-batch per ≤ 100. L4 satu panggilan per graph (murah, tanpa batching). Gleaning menggandakan panggilan ekstraksi direct. |
| Auditability | 4 | Fusi sangat lengkap (stub, ignored decisions, konflik, varian). Saat konversi lebih tipis: L4 hanya statistik, L3 forensik hanya `--debug`, konflik registry tidak dicatat. |
| Kecocokan arsitektur dengan hybrid_rag | 2 | Schema-first + networkx per dokumen vs open-schema Neo4j inkremental per chunk. |
| Kematangan kode | 4 | Guard didorong insiden nyata yang terdokumentasi di komentar, ada parity test identitas. Ada kode mati (`identity_fields_map`, `merge_similarity_fallback`) dan docstring usang. |

**Kesimpulan:** docling-graph unggul pada **disiplin lapisan alias**: satu fungsi identitas yang konsisten, LLM dibatasi proposer dan veto deterministik, audit fusi yang lengkap, dan keputusan manusia yang bisa di-replay. Kelemahannya ada di dua ujung. Lapisan dasar (registry exact) menggabungkan homonim, ID kosong, dan nilai non-Latin tanpa jejak. Lapisan atas (veto shared-neighbor, co-occurrence batch, contagion lintas dokumen) cukup ketat sehingga banyak alias sah tidak pernah tergabung.

Untuk hybrid_rag, yang diambil adalah **polanya**: kunci kanonik tunggal, digit guard, LLM yang dibatasi proposer, batching id-space, decision record yang bisa di-replay, dan konflik nilai sebagai sinyal. Yang **tidak boleh disalin mentah**:
- veto co-occurrence per batch (memblokir definisi singkatan);
- veto shared-child (relasi yang sama adalah bukti alias, bukan bukti sibling);
- contagion konflik per kelas (perlu daftar field identitas-kritis, bukan semua scalar);
- identitas tanpa scope parent dan tanpa pencatatan konflik;
- normalisasi yang bergantung pada nama field.

Yang harus ditambahkan sendiri: proposer yang lebih kaya (token-bag, akronim, embedding), alias registry persisten yang dipakai ulang sebagai identitas, dan konteks deskripsi untuk LLM.
