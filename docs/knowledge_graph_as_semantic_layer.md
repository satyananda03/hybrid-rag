## Knowledge Graph as Semantic Layer for Database 

**Deskripsi kolom** bersifat **lokal (1-to-1)** dan menjelaskan *apa isi dari satu kolom fisik*, sedangkan node **`BusinessConcept` / `Metric`** bersifat **komposit (1-to-Many / Cross-Table)** dan mendefinisikan *aturan kalkulasi atau logika bisnis* yang melibatkan banyak kolom sekaligus.

| **Dimensi** | **Deskripsi pada Node Column** | **Node BusinessConcept / Metric** |
| --- | --- | --- |
| **Cakupan (Scope)** | **1 Kolom Fisik** (Lokal pada satu tabel). | **Multi-Kolom & Multi-Tabel** (Komposit). |
| **Fungsi Utama** | Menjelaskan arti teknis/semantik dari data mentah yang tersimpan di kolom tersebut. | Menjelaskan **rumus SQL**, **filter wajib (`WHERE`)**, dan **aturan agregasi** dari suatu istilah bisnis. |
| **Contoh Isi** | `gross_amount`: *"Nilai kotor transaksi dalam Rupiah sebelum dipotong diskon dan pajak."* | **Pendapatan Bersih (Net Revenue)**: *"Total `gross_amount` dikurangi `discount` dan `tax`, khusus untuk transaksi dengan `status = 'SETTLED'` dan bukan transaksi `is_refunded = true`."* |
| **Bentuk di KG** | Properti string biasa di dalam node `Column` (`col.description`). | **Node tersendiri** yang memiliki relasi `[:DEPENDS_ON]` atau `[:USES_COLUMN]` ke beberapa node `Column`. |
|  |  |  |

### 2. Mengapa Tidak Cukup Hanya Mengandalkan Deskripsi Kolom?

Bayangkan user bertanya ke SQL Agent:

> *"Berapa **Rasio Kepatuhan Audit** di wilayah Jawa Timur bulan lalu?"*
> 

Di dalam database fisik, **tidak ada** kolom bernama `rasio_kepatuhan_audit`. Yang ada hanyalah:

- Tabel `submissions` (kolom `id`, `status_verifikasi`, `created_at`)
- Tabel `instansi` (kolom `id`, `kategori_wajib_lapor`, `kode_provinsi`)

Jika Anda hanya mengandalkan **deskripsi kolom** dan melakukan *Vector Search* atas kalimat *"Rasio Kepatuhan Audit"*, inilah yang akan terjadi:

1. **Kegagalan Retrieval Komposit:** Vector search mungkin menemukan kolom `status_verifikasi`, tetapi gagal menarik kolom `kategori_wajib_lapor` di tabel `instansi` karena secara semantik kata "wajib lapor" kurang mirip dengan "kepatuhan audit". Akibatnya, pembagi (*denominator*) rumus rasio tersebut hilang.
2. **Halusinasi Rumus Matematika:** Sekalipun LLM berhasil mendapatkan kedua tabel, LLM tidak tahu aturan bisnis internal organisasi Anda: apakah `status_verifikasi` yang dihitung itu yang bernilai `'APPROVED'` saja atau termasuk `'SUBMITTED'`? Apakah instansi yang tidak aktif ikut menjadi pembagi?
3. **Duplikasi & Inkonsistensi (DRY Principle):** Jika Anda memaksakan menulis rumus *"Rasio Kepatuhan Audit"* di dalam deskripsi kolom `status_verifikasi`, lalu besok ada 10 metrik bisnis lain yang juga menggunakan kolom `status_verifikasi`, deskripsi kolom tersebut akan bengkak menjadi ribuan karakter dan saling tumpang tindih.

### 3. Bagaimana Node `BusinessConcept` Bekerja di Knowledge Graph?

Dengan menjadikan `BusinessConcept` atau `Metric` sebagai **Node kelas satu (First-Class Citizen)** di dalam graph, struktur penarikan konteks (retrieval) menjadi deterministik:

#### A. Struktur Properti Node `BusinessConcept`

```
{
  "id": "metric_net_revenue",
  "name": "Pendapatan Bersih",
  "synonyms": ["Net Revenue", "Omzet Bersih", "Net Sales"],
  "description": "Total nilai transaksi sukses setelah dikurangi diskon dan biaya admin.",
  "sql_template": "SUM(t.gross_amount - COALESCE(t.discount, 0) - COALESCE(p.admin_fee, 0))",
  "mandatory_filters": [
    "t.status = 'SETTLED'",
    "t.deleted_at IS NULL"
  ]
}
```

#### B. Relasi (Edges) di dalam Graph

Node `metric_net_revenue` dihubungkan menggunakan *edge* `[:USES_COLUMN]` ke:

- `(:Column {name: "orders.gross_amount"})`
- `(:Column {name: "orders.discount"})`
- `(:Column {name: "orders.status"})`
- `(:Column {name: "orders.deleted_at"})`
- `(:Column {name: "payments.admin_fee"})`

#### C. Efeknya saat Query Masuk (Graph Traversal)

1. User bertanya tentang *"Omzet Bersih"*.
2. Hybrid search langsung mengenai node `(:BusinessConcept {name: "Pendapatan Bersih"})` lewat kecocokan sinonim.
3. Melalui **1-hop graph traversal** (`MATCH (m:BusinessConcept)-[:USES_COLUMN]->(c:Column)<-[:HAS_COLUMN]-(t:Table)`), sistem **otomatis menarik seluruh kolom dan tabel penyusunnya** (`orders` dan `payments`), meskipun nama kolom seperti `deleted_at` atau `admin_fee` sama sekali tidak disebut oleh user.
4. Sistem juga langsung menemukan jalur `JOIN` antara tabel `orders` dan `payments`, serta menyuntikkan `sql_template` dan `mandatory_filters` ke dalam prompt SQL Agent.

**Kapan Cukup Pakai Deskripsi Kolom, dan Kapan Harus Buat `BusinessConcept`?**
Agar pembuatan KG tetap efisien dan tidak *over-engineering*, gunakan aturan praktis berikut:
• **Cukup tulis di Deskripsi Kolom (`Column.description`) jika:**
    ◦ Konsep tersebut memang hanya diwakili oleh **satu kolom tunggal** tanpa syarat filter tersembunyi (contoh: "Tanggal Lahir" $\rightarrow$ langsung merujuk ke `users.birth_date`).
    ◦ Hanya menjelaskan arti kode atau satuan pada kolom tersebut (contoh: *"Berat barang dalam satuan kilogram (kg)"*).
• **Wajib buat Node `BusinessConcept` terpisah jika:**
    ◦ **Melibatkan >1 kolom** (penjumlahan, pengurangan, rasio/persentase lintas kolom atau lintas tabel).
    ◦ **Memiliki syarat filter implisit (`WHERE`)** yang tidak pernah diucapkan user (contoh: "User Aktif" artinya `status = 'ACTIVE' AND last_login >= NOW() - INTERVAL '30 days'`).
    ◦ **Merupakan KPI / Jargon Organisasi** (contoh: *Churn Rate*, *NPL*, * Skor Kematangan*, *Realisasi Anggaran*, *LTV*) yang rumusnya harus baku dan tidak boleh dikarang bebas oleh LLM.

Kenapa Butuh BussinessConcept tidak hanya Description per column atau per table/schema? bukankah Bussiness Concept Metrics harusnya dibuatkan kolom khusus saat data warehousing?

Jawaban :

Jika kita bisa memindahkan beban logika bisnis ke hulu (ke dalam pipeline ETL/ELT Data Warehouse seperti `dbt` atau *Materialized View* di PostgreSQL), maka beban kognitif LLM saat menulis SQL di hilir akan turun drastis, dan risiko salah rumus hampir nol.

Namun, **tidak semua metrik bisnis bisa dijadikan kolom fisik di Data Warehouse.** Di dalam dunia *Analytics Engineering*, ada batas tegas mengenai **apa yang wajib masuk ke Data Warehouse (Physical Column)** dan **apa yang wajib tetap berada di Semantic Layer / Knowledge Graph (`sql_template`)**.

### 1. Kapan Ide Anda Benar? (Row-Level & Additive Metrics $\rightarrow$ Masuk Data Warehouse)

Untuk contoh `net_revenue` pada tingkat per-transaksi (`gross_amount - discount - admin_fee`), ide Anda adalah **best practice terbaik**.

Di dalam arsitektur *Medallion* (Bronze $\rightarrow$ Silver $\rightarrow$ Gold) atau *Star Schema*:

- Saat pipeline ETL/ELT berjalan (atau lewat fitur `GENERATED ALWAYS AS` / `VIEW` di PostgreSQL), kita langsung membuat kolom fisik bernama `net_amount` di tabel `fact_transactions`:SQL
    
    ```
    -- Dilakukan di pipeline Data Warehouse / dbt / View:
    (t.gross_amount - COALESCE(t.discount, 0) - COALESCE(p.admin_fee, 0)) AS net_amount
    ```
    
- Bahkan untuk `mandatory_filters` (`status = 'SETTLED' AND deleted_at IS NULL`), di lapisan **Gold / Data Mart**, baris yang belum *settled* atau sudah dihapus biasanya langsung dibuang atau dipisahkan ke tabel khusus `fact_settled_orders`.
- **Hasilnya bagi SQL Agent:** Agent tidak perlu lagi melakukan `JOIN` ke tabel `payments` untuk mencari `admin_fee` atau merakit rumus pengurangan. Agent cukup membaca `Column.description` pada kolom `net_amount` dan menulis:SQL
    
    ```
    SELECT SUM(net_amount) FROM fact_settled_orders;
    ```
    

### 2. Kapan Tidak Bisa Dijadikan Kolom di Data Warehouse? (Mengapa `sql_template` di KG Tetap Dibutuhkan)

Ada **4 kondisi nyata** di mana pembuatan kolom fisik di Data Warehouse **mustahil dilakukan atau justru menghasilkan angka yang salah fatal**:

#### A. Jebakan Metrik Non-Aditif (Rasio, Persentase, dan Pembagian)

Ini adalah alasan nomor satu mengapa *Semantic Layer* (dan node `BusinessConcept`) diciptakan.

Bayangkan metrik: **Margin Laba Bersih (%)** = `SUM(net_profit) / SUM(gross_revenue)`, atau **Persentase Realisasi Anggaran** = `SUM(realisasi) / SUM(pagu_anggaran)`.

Jika Anda menghitung persentase tersebut di Data Warehouse sebagai kolom per-baris (`margin_pct_row`), lalu user bertanya: *"Berapa Margin Laba Bersih untuk wilayah Jawa Timur?"*, SQL Agent yang naif akan menjumlahkan atau merata-rata kolom tersebut:

- `SUM(margin_pct_row)` $\rightarrow$ **Salah total** (persentase tidak bisa dijumlahkan: 50% + 50% bukan 100%).
- `AVG(margin_pct_row)` $\rightarrow$ **Salah secara matematis (*Simpson's Paradox / Average of Averages*)**, karena transaksi Rp 10.000 yang marginnya 90% dianggap punya bobot yang sama dengan transaksi Rp 1 Miliar yang marginnya 5%.

Untuk seluruh metrik berbentuk **Rasio ($A / B$)**, penjumlahan wajib dilakukan **terlebih dahulu** sesuai `GROUP BY` dan `WHERE` yang diminta user saat *runtime*, baru kemudian dibagi:

$$\text{Rasio Kepatuhan / Realisasi} = \frac{\sum \text{Pembilang}}{\sum \text{Penyebut}}$$

Rumus agregasi dinamis seperti `SUM(realisasi) / NULLIF(SUM(pagu), 0)` ini **tidak bisa disimpan sebagai kolom baris**, sehingga wajib disimpan sebagai `sql_template` di Knowledge Graph / Semantic Layer.

#### B. Metrik `COUNT(DISTINCT ...)` Lintas Dimensi Waktu & Kategori

Contoh: **Jumlah User Aktif (MAU)** atau **Jumlah Instansi yang Sudah Lapor (`COUNT(DISTINCT instansi_id)`)**.

Anda tidak bisa membuat tabel pre-agregasi di Data Warehouse yang menyimpan jumlah `COUNT(DISTINCT)` lalu menjumlahkannya kembali.

- Jika di Bulan Januari ada 100 user unik, dan di Bulan Februari ada 100 user unik (di mana 80 orang adalah orang yang sama), maka total user unik selama 2 bulan adalah **120 orang**, bukan `100 + 100 = 200`.
- Jika user meminta filter dinamis (*"Hitung user unik antara tanggal 5 Maret sampai 18 April khusus kategori UMKM"*), query `COUNT(DISTINCT user_id)` harus dieksekusi langsung di atas data granular saat *runtime*.

#### C. Aturan Filter Kontekstual (Conditional Aggregation)

Dalam satu tabel yang sama, definisi "status valid" bisa berbeda tergantung metrik apa yang ditanyakan user:

- Jika user tanya **"Total Penjualan (GMV)"** $\rightarrow$ hitung yang `status IN ('PAID', 'SHIPPED', 'COMPLETED')`.
- Jika user tanya **"Tingkat Pembatalan (Cancellation Rate)"** $\rightarrow$ hitung `COUNT(*) FILTER (WHERE status = 'CANCELLED') / COUNT(*)`.
    
    Jika di Data Warehouse Anda sudah membuang semua baris yang `status = 'CANCELLED'`, maka Agent tidak akan pernah bisa menjawab pertanyaan tentang *Cancellation Rate*.
    

#### D. Akses ke Database Operasional (OLTP) atau Database Pihak Ketiga (Read-Only)

Sering kali AI Agent ditugaskan membaca database aplikasi berjalan (OLTP) atau *replica* database instansi di mana Anda **tidak punya izin untuk mengubah skema tabel** atau membangun pipa ETL Data Warehouse yang berat. Dalam kondisi ini, Knowledge Graph bertindak sebagai **"Virtual Data Warehouse"** yang merapikan skema berantakan tanpa menyentuh tabel fisik.

### 3. Best Practices: Pembagian Tugas Data Warehouse vs. Knowledge Graph

**Best Practices: Pembagian Tugas "Gold Layer (Warehouse)" vs. "Semantic Layer (KG)"**.

Let's write a crisp, authoritative explanation and include the interactive widget at the end.Insting *Data Engineering* Anda **100% tepat**. Untuk kasus aritmatika level baris (*row-level calculation*) seperti `gross_amount - discount - admin_fee`, memindahkannya ke pipeline **Data Warehouse (Gold/Mart Layer)** atau **Materialized View** sebagai kolom fisik (misal: `net_amount`) adalah langkah paling aman, cepat, dan sangat direkomendasikan.

Semakin sederhana SQL yang harus ditulis oleh LLM (cukup `SUM(net_amount)` daripada merakit `SUM(t.gross - COALESCE(...) - ...)` lintas tabel), semakin rendah risiko halusinasi dan error sintaksnya.

Namun, mengapa di arsitektur *enterprise* (termasuk di tool seperti dbt Semantic Layer, Cube, atau Knowledge Graph SQL Agent) kita **tetap tidak bisa menghapus konsep `BusinessMetric` / Semantic Layer** dan mengganti semuanya menjadi kolom fisik di Data Warehouse?

Ada **3 batasan matematis dan arsitektural** di mana kolom Data Warehouse tidak bisa menggantikan peran KG/Semantic Layer:

### 1. Jebakan Matematika pada Metrik Non-Aditif (*The Ratio & Distinct Trap*)

Dalam dunia Data Warehousing (Kimball Dimensional Modeling), metrik dibagi menjadi tiga: **Additive**, **Semi-Additive**, dan **Non-Additive**.

- **Metrik Aditif (Contoh: `net_revenue`):** Bisa dihitung per baris (`gross - discount - fee`), disimpan sebagai kolom baru, lalu dijumlahkan (`SUM`) berdasarkan dimensi apa pun (per hari, per cabang, per provinsi). **Ini wajib masuk ke Data Warehouse.**
- **Metrik Non-Aditif (Contoh: Rasio, Persentase, Margin, Skor Rata-rata Tertimbang):** **TIDAK BISA** dihitung di level baris lalu di-`SUM` atau di-`AVG` begitu saja oleh Agent.

#### Contoh Kasus Nyata (Mengapa Kolom Pre-computed Gagal pada Rasio):

Bayangkan Anda memiliki tabel anggaran dengan kolom `pagu_anggaran` dan `realisasi`. Anda ingin memudahkan Agent menghitung **"Persentase Serapan Anggaran"**, lalu di Data Warehouse Anda buatkan kolom baru per baris kegiatan:
`persen_serapan = (realisasi / pagu_anggaran) * 100`

### 2. Keterbatasan Akses ke Database Operasional (OLTP / Legacy DB)

Dalam praktik di lapangan:

- Jika Anda memiliki **kendali penuh atas Data Warehouse / dbt / Airflow**, Anda bebas membuat tabel *Data Mart* atau *Materialized View* baru.
- Namun, sering kali SQL Agent dipasang langsung di atas **Read-Replica Database Produksi (OLTP)** atau database aplikasi klien/pemerintah di mana tim AI **dilarang mengubah skema tabel (`ALTER TABLE`)** atau menambah beban ETL baru. Dalam kondisi *schema read-only* seperti ini, Knowledge Graph / Semantic Layer berfungsi sebagai **"Virtual Data Mart"** tanpa perlu menyentuh database fisik.

### Best Practices: Pembagian Tugas antara Data Warehouse vs. Knowledge Graph (Hybrid Architecture)

Agar arsitektur Anda paling efisien, cepat, dan mudah di-*maintain*, jangan pilih salah satu—terapkan **Aturan Pembagian Lapisan (Push-Down Rule)** berikut:

| **Jenis Logika / Transformasi** | **Tempat Terbaik (Best Practice)** | **Alasan Teknis & Contoh** |
| --- | --- | --- |
| **1. Pembersihan Data & Konversi Tipe** (`COALESCE`, casting tipe data, standarisasi tanggal/kode wilayah) | **Data Warehouse / ETL Pipeline (Silver Layer)** | Jangan biarkan LLM menulis `CAST(REGEXP_REPLACE(...) AS NUMERIC)` saat runtime. Bersihkan di ETL. |
| **2. Aritmatika Level Baris & Pre-Join Umum** (Metrik Aditif seperti `gross - discount - fee`) | **Data Warehouse / Materialized View (Gold Layer)** | Buat kolom fisik `net_amount` atau `GENERATED ALWAYS AS (...) STORED` di PostgreSQL. Agent tinggal `SUM(net_amount)` + baca deskripsi kolomnya. |
| **3. Metrik Non-Aditif (Rasio, Persentase, `COUNT DISTINCT`, Pertumbuhan YoY/MoM)** | **Knowledge Graph / Semantic Layer (`BusinessConcept`)** | Karena pembilang dan penyebut harus di-`SUM` terlebih dahulu sesuai `GROUP BY` yang diminta user secara dinamis ($\frac{\sum A}{\sum B}$). |
| **4. Istilah Bisnis Komposit & Aturan Filter Dinamis** (Misal: *"Instansi Kritis"*, *"Vendor Bermasalah"*) | **Knowledge Graph / Semantic Layer (`BusinessConcept`)** | Definisi bisnis sering berubah (misal ambang batas skor dari $< 60$ diubah jadi $< 70$). Mengubah 1 node di KG butuh 1 detik tanpa perlu *backfill* ulang jutaan baris di Data Warehouse.