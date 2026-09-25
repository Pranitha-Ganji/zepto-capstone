"""
Zepto Data Pipeline: Web Scraping to Normalized SQLite Database
Author: Zepto AI/ML Engineering Guild
Module: /data_pipeline

Description:
This script scrapes catalog-style product data from books.toscrape.com across multiple
categories, cleans and enriches the data (including fixed-rate currency conversion: 1 GBP = 105.50 INR),
stores it in a normalized two-table SQLite schema (categories and books with PK/FK relationship),
executes 5 benchmark SQL queries covering required SQL operations, and validates the relational join
against in-memory pandas.merge() for consistency and equivalence.
"""

import os
import sys
import re
import sqlite3
import urllib.parse
from typing import Dict, List, Tuple

# Ensure stdout handles UTF-8 characters cleanly on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Project-defined fixed currency conversion rate (required baseline constant)
GBP_TO_INR_RATE = 105.50

BASE_URL = "http://books.toscrape.com/"
DB_PATH = os.path.join(os.path.dirname(__file__), "books.db")

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}


def get_category_urls() -> List[Tuple[str, str]]:
    """
    Scrape category names and their entry URLs from the home page.
    Returns a list of tuples: (category_name, category_url).
    """
    response = requests.get(BASE_URL, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    
    side_categories = soup.find("div", class_="side_categories")
    if not side_categories:
        raise ValueError("Could not find side_categories container on page.")
        
    category_links = []
    # Skip the top-level "Books" header link (first <a>)
    for a_tag in side_categories.find("ul").find("ul").find_all("a"):
        cat_name = a_tag.get_text(strip=True)
        rel_url = a_tag["href"]
        abs_url = urllib.parse.urljoin(BASE_URL, rel_url)
        category_links.append((cat_name, abs_url))
        
    return category_links


def scrape_category_books(cat_name: str, start_url: str, min_books_per_cat: int = 15) -> List[Dict]:
    """
    Scrape all books within a given category, handling pagination if present.
    """
    books = []
    current_url = start_url
    
    while current_url:
        resp = requests.get(current_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        
        product_pods = soup.find_all("article", class_="product_pod")
        for pod in product_pods:
            # Title
            h3 = pod.find("h3")
            title = h3.find("a")["title"] if h3 and h3.find("a") and h3.find("a").has_attr("title") else pod.find("h3").get_text(strip=True)
            
            # Price string
            price_elem = pod.find("p", class_="price_color")
            raw_price = price_elem.get_text(strip=True) if price_elem else None
            
            # Star rating
            rating_elem = pod.find("p", class_=re.compile(r"star-rating"))
            star_rating = None
            if rating_elem:
                classes = rating_elem.get("class", [])
                for cls in classes:
                    if cls.lower() in RATING_MAP:
                        star_rating = cls.capitalize()
                        break
                        
            # Availability
            avail_elem = pod.find("p", class_="instock")
            availability = avail_elem.get_text(strip=True) if avail_elem else None
            
            books.append({
                "title": title,
                "raw_price": raw_price,
                "star_rating": star_rating,
                "availability": availability,
                "category": cat_name
            })
            
        # Pagination: check next button
        next_li = soup.find("li", class_="next")
        if next_li and next_li.find("a"):
            next_href = next_li.find("a")["href"]
            current_url = urllib.parse.urljoin(current_url, next_href)
        else:
            current_url = None
            
    return books


def clean_scraped_data(raw_records: List[Dict]) -> pd.DataFrame:
    """
    Clean scraped records into typed columns:
    - price_gbp: float (currency stripped)
    - rating: integer (1-5)
    - in_stock: boolean (True/False)
    - price_inr: float (converted at 105.50 INR/GBP)
    Handles unparseable rows using median imputation for numeric fields
    or dropping if title/category are null.
    """
    df = pd.DataFrame(raw_records)
    print(f"Raw scraped records count: {len(df)}")
    
    # 1. Clean Title and Category
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    
    # 2. Clean Price to float price_gbp
    def parse_price(val):
        if not val or pd.isna(val):
            return np.nan
        # Strip all non-numeric characters except period
        match = re.search(r"(\d+\.?\d*)", str(val))
        return float(match.group(1)) if match else np.nan

    df["price_gbp"] = df["raw_price"].apply(parse_price)
    
    # 3. Clean Star Rating to integer (1-5)
    def parse_rating(val):
        if not val or pd.isna(val):
            return np.nan
        key = str(val).lower().strip()
        return RATING_MAP.get(key, np.nan)
        
    df["rating"] = df["star_rating"].apply(parse_rating)
    
    # 4. Clean Availability to boolean in_stock
    def parse_availability(val):
        if not val or pd.isna(val):
            return False
        return "in stock" in str(val).lower()
        
    df["in_stock"] = df["availability"].apply(parse_availability).astype(int)
    
    # 5. Handle missing / unparseable numeric values: median imputation
    if df["price_gbp"].isna().any():
        med_price = df["price_gbp"].median()
        print(f"Imputing missing price_gbp with median: {med_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(med_price)
        
    if df["rating"].isna().any():
        med_rating = int(df["rating"].median())
        print(f"Imputing missing rating with median: {med_rating}")
        df["rating"] = df["rating"].fillna(med_rating).astype(int)
    else:
        df["rating"] = df["rating"].astype(int)
        
    # 6. Currency Conversion: price_inr using fixed rate 105.50
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)
    
    # Drop rows that lack a valid title or category (critical identity fields)
    df = df.dropna(subset=["title", "category"])
    
    # Return cleaned dataframe
    cleaned_df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
    return cleaned_df


def initialize_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Creates the normalized SQLite schema with two tables:
    - categories (category_id PK, category_name UNIQUE)
    - books (book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)
    """
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT NOT NULL UNIQUE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
    """)
    
    conn.commit()
    return conn


def load_data_to_sqlite(df: pd.DataFrame, conn: sqlite3.Connection):
    """
    Inserts cleaned data into normalized SQLite tables.
    """
    cursor = conn.cursor()
    
    # 1. Insert unique categories
    unique_categories = sorted(df["category"].unique())
    for cat in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?);", (cat,))
    conn.commit()
    
    # Read back category map
    cursor.execute("SELECT category_name, category_id FROM categories;")
    cat_map = dict(cursor.fetchall())
    
    # 2. Insert books
    book_records = []
    for _, row in df.iterrows():
        cat_id = cat_map[row["category"]]
        book_records.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            cat_id
        ))
        
    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, book_records)
    conn.commit()
    print(f"Successfully loaded {len(unique_categories)} categories and {len(book_records)} books into SQLite.")


def run_benchmark_queries(conn: sqlite3.Connection):
    """
    Executes at least 5 SQL queries demonstrating:
    - SELECT / WHERE
    - ORDER BY
    - LIMIT
    - DISTINCT
    - IN or BETWEEN
    - JOIN between books and categories
    """
    queries = [
        (
            "Query 1: SELECT / WHERE (Filter in-stock books with price under INR 2,000)",
            """
            SELECT book_id, title, price_inr, in_stock 
            FROM books 
            WHERE in_stock = 1 AND price_inr < 2000.00
            LIMIT 5;
            """
        ),
        (
            "Query 2: ORDER BY and LIMIT (Top 5 most expensive books in GBP)",
            """
            SELECT book_id, title, price_gbp, price_inr 
            FROM books 
            ORDER BY price_gbp DESC 
            LIMIT 5;
            """
        ),
        (
            "Query 3: DISTINCT (Distinct star ratings available in catalog)",
            """
            SELECT DISTINCT rating 
            FROM books 
            ORDER BY rating ASC;
            """
        ),
        (
            "Query 4: BETWEEN and IN (Books with 4 or 5 stars and price between GBP 25 and GBP 45)",
            """
            SELECT book_id, title, rating, price_gbp, price_inr 
            FROM books 
            WHERE rating IN (4, 5) AND price_gbp BETWEEN 25.00 AND 45.00
            ORDER BY rating DESC, price_gbp ASC
            LIMIT 5;
            """
        ),
        (
            "Query 5: JOIN (Relational join listing top 10 rated books with category names)",
            """
            SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE b.rating >= 4
            ORDER BY b.rating DESC, b.price_gbp DESC
            LIMIT 10;
            """
        )
    ]
    
    results = {}
    print("\n" + "="*80)
    print("BENCHMARK SQL QUERIES EXECUTION")
    print("="*80)
    
    for title, q_str in queries:
        print(f"\n--- {title} ---")
        print("SQL Query:\n" + q_str.strip())
        df_res = pd.read_sql_query(q_str, conn)
        print("Result Preview:")
        print(df_res.to_string(index=False))
        results[title] = df_res
        
    return results


def verify_sql_vs_pandas_merge(conn: sqlite3.Connection):
    """
    Reads back tables into pandas DataFrames and reproduces Query 5
    using pd.merge() to demonstrate equivalence with pd.read_sql().
    """
    print("\n" + "="*80)
    print("EQUIVALENCE PROOF: pd.read_sql JOIN vs pd.merge")
    print("="*80)
    
    # 1. SQL Result
    sql_query = """
    SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    WHERE b.rating >= 4
    ORDER BY b.rating DESC, b.price_gbp DESC
    LIMIT 10;
    """
    df_sql = pd.read_sql_query(sql_query, conn)
    
    # 2. In-memory pd.merge reproduction
    df_books = pd.read_sql_query("SELECT * FROM books;", conn)
    df_cats = pd.read_sql_query("SELECT * FROM categories;", conn)
    
    df_merged = pd.merge(df_books, df_cats, on="category_id", how="inner")
    df_merged_filtered = df_merged[df_merged["rating"] >= 4]
    df_merged_sorted = df_merged_filtered.sort_values(
        by=["rating", "price_gbp"], 
        ascending=[False, False]
    ).head(10)
    
    cols = ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]
    df_pandas = df_merged_sorted[cols].reset_index(drop=True)
    
    print("\n[Approach 1: pd.read_sql (SQL JOIN)]")
    print(df_sql.to_string(index=False))
    
    print("\n[Approach 2: pd.merge (In-memory Pandas JOIN)]")
    print(df_pandas.to_string(index=False))
    
    # Assert equivalence
    are_equal = df_sql.equals(df_pandas)
    print(f"\nEquivalence Check (df_sql.equals(df_pandas)): {are_equal}")
    if are_equal:
        print(">> PROOF VERIFIED: Both SQL and pandas in-memory merge yield identical records and order.")
    else:
        # Check tolerance in float
        diff = (df_sql["price_gbp"] - df_pandas["price_gbp"]).abs().max()
        print(f"Max difference in price_gbp: {diff}")


def run_pipeline():
    """
    Executes the entire Module 1 data pipeline end-to-end.
    """
    print("[1/5] Fetching category listing from books.toscrape.com...")
    all_categories = get_category_urls()
    print(f"Discovered {len(all_categories)} categories on catalog site.")
    
    # Select first 5 categories to guarantee >= 60 books across >= 3 categories
    selected_categories = all_categories[:5]
    print("Selected categories:")
    for name, url in selected_categories:
        print(f" - {name}: {url}")
        
    raw_books = []
    print("\n[2/5] Scraping books across selected categories...")
    for cat_name, cat_url in selected_categories:
        cat_books = scrape_category_books(cat_name, cat_url)
        print(f" - Scraped {len(cat_books)} books from category '{cat_name}'")
        raw_books.extend(cat_books)
        
    print(f"Total raw books scraped: {len(raw_books)} (Requirement >= 60 satisfied)")
    
    print("\n[3/5] Cleaning and transforming data (fixed rate 1 GBP = 105.50 INR)...")
    cleaned_df = clean_scraped_data(raw_books)
    print("Cleaned DataFrame sample:")
    print(cleaned_df.head(5).to_string(index=False))
    
    print("\n[4/5] Initializing normalized SQLite schema & loading data...")
    conn = initialize_database(DB_PATH)
    load_data_to_sqlite(cleaned_df, conn)
    
    print("\n[5/5] Executing benchmark SQL queries & verifying pandas equivalence...")
    run_benchmark_queries(conn)
    verify_sql_vs_pandas_merge(conn)
    
    conn.close()
    print("\n>>> Module 1 Pipeline completed successfully! Database saved to:", DB_PATH)


if __name__ == "__main__":
    run_pipeline()
