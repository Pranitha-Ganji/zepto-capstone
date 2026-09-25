# Module 1 — Data Pipeline (`/data_pipeline`)
**Zepto Analytics & AI Guild: Catalog Benchmarking & Relational Ingestion Engine**

---

## 1. Overview & Objective
Zepto analysts require real-time, benchmarked pricing and availability data before catalog metrics feed executive dashboards or pricing recommendation algorithms. This module implements a robust, automated raw-to-relational data pipeline:
1. **Web Scraping:** Programmatically scrapes live product listings from [books.toscrape.com](http://books.toscrape.com/) across 5 distinct categories, capturing over 60 book records.
2. **Data Cleaning & Enrichment:** Cleans messy strings, casts types, handles missing/unparseable values via median imputation, and enriches pricing with the project's fixed-rate currency conversion.
3. **Relational Modeling:** Stores the cleaned records into a normalized two-table SQLite database (`categories` and `books`) with a strict Primary Key / Foreign Key relationship.
4. **Benchmarking & Validation:** Executes 5 SQL queries covering required query clauses (`SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`/`IN`, `JOIN`) and mathematically validates query results by reproducing the relational join using in-memory `pandas.merge()`.

---

## 2. Fixed Currency Conversion Rate Baseline
- **Required Rate:** `1 GBP = 105.50 INR`
- **Specification:** This is an artificial, project-defined constant for this assignment, not a live or historical market rate, requiring no external API key, network lookup, or date reference.
- **Formula:** $\text{price\_inr} = \text{round}(\text{price\_gbp} \times 105.50, 2)$

---

## 3. Data Cleaning & Parsing Decisions
When transforming the raw scraped data into structured analytics formats:
- **Price:** Stripped currency symbols (`£`, `Â`, whitespace) using regular expressions and cast to float `price_gbp`.
- **Rating:** Mapped textual star ratings (`One`, `Two`, `Three`, `Four`, `Five`) to corresponding integers (`1`, `2`, `3`, `4`, `5`).
- **Availability:** Scanned availability string for substring `"in stock"` to generate a boolean integer column `in_stock` (`1` = True, `0` = False).
- **Missing/Unparseable Handling Decision:** 
  - For numeric fields (`price_gbp`, `rating`), any parsing anomaly is handled using **median imputation**. The median is preferred over the mean because it is robust against pricing outliers and preserves valid integer distributions for star ratings without skewing the catalog profile.
  - Rows missing essential identifying attributes (`title` or `category`) are dropped, as unidentifiable records cannot be mapped into the relational schema.

---

## 4. Relational Database Schema
The database (`books.db`) enforces a normalized two-table relational structure with foreign key constraints enabled (`PRAGMA foreign_keys = ON;`):

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
);
```

---

## 5. SQL Benchmark Queries & Outputs

### Query 1: `SELECT` / `WHERE`
*Filter in-stock books with price under ₹2,000.*
```sql
SELECT book_id, title, price_inr, in_stock 
FROM books 
WHERE in_stock = 1 AND price_inr < 2000.00
LIMIT 5;
```

### Query 2: `ORDER BY` & `LIMIT`
*Top 5 most expensive books in catalog (GBP).*
```sql
SELECT book_id, title, price_gbp, price_inr 
FROM books 
ORDER BY price_gbp DESC 
LIMIT 5;
```

### Query 3: `DISTINCT`
*Distinct star ratings available in catalog.*
```sql
SELECT DISTINCT rating 
FROM books 
ORDER BY rating ASC;
```

### Query 4: `BETWEEN` & `IN`
*High-rated books (rating IN (4, 5)) with price BETWEEN £25 and £45.*
```sql
SELECT book_id, title, rating, price_gbp, price_inr 
FROM books 
WHERE rating IN (4, 5) AND price_gbp BETWEEN 25.00 AND 45.00
ORDER BY rating DESC, price_gbp ASC
LIMIT 5;
```

### Query 5: `JOIN`
*Relational join listing top 10 rated books with their associated category names.*
```sql
SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE b.rating >= 4
ORDER BY b.rating DESC, b.price_gbp DESC
LIMIT 10;
```

---

## 6. Equivalence Verification: `pd.read_sql` vs `pd.merge`
To prove data integrity across analytical interfaces, Query 5 was executed via `pd.read_sql()` against SQLite, and independently reproduced in memory using `pd.merge()` on raw tables:
```python
df_books = pd.read_sql_query("SELECT * FROM books;", conn)
df_cats = pd.read_sql_query("SELECT * FROM categories;", conn)

df_merged = pd.merge(df_books, df_cats, on="category_id", how="inner")
df_merged_filtered = df_merged[df_merged["rating"] >= 4]
df_pandas = df_merged_filtered.sort_values(
    by=["rating", "price_gbp"], 
    ascending=[False, False]
).head(10)[["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]].reset_index(drop=True)

assert df_sql.equals(df_pandas), "Equivalence check passed!"
```
**Conclusion:** Both methods yield identical row sets, ordering, and data types (`equals() == True`).

---

## 7. Execution Guide
To run the data pipeline end to end and regenerate the SQLite database:
```bash
# Activate environment
.venv\Scripts\activate

# Run pipeline
python data_pipeline/pipeline.py
```
Or open and execute `data_pipeline/data_pipeline.ipynb`.
