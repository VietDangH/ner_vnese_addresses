"""Dataset 4 - masothue.com official business registrations.

Two roles in one file:

1. **Crawl** (`--crawl`): scrape ~`target_size` addresses per province from
   masothue.com (a tax-code lookup directory) using curl_cffi (Chrome
   impersonation) + BeautifulSoup, writing the raw `addresses_50_per_city.csv`.
   Ported from Crawl_Data.ipynb; needs `curl_cffi` and `beautifulsoup4`+`lxml`.

2. **Preprocess** (default): the registrations are already standardised, so per
   the report they need only the shared pass. The raw rows carry a trailing
   ", Việt Nam" segment, so we drop that last comma-segment (as the notebook's
   Data_Preprocessing does), then run the shared pass.

    load `address`  ->  drop trailing country segment  ->  normalize + de-dup
    ->  EDA  ->  segment  ->  number fix-ups  ->  masothue_final.csv

Run from the repo root (`_organized/`):

    python -m data_processing.crawl_masothue                     # preprocess raw file
    python -m data_processing.crawl_masothue --crawl --per-province 50
    python -m data_processing.crawl_masothue --no-strip-country --no-segment
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

BASE_URL = "https://masothue.com"


# --------------------------------------------------------------------------- #
# Crawl  (imports curl_cffi / bs4 lazily so preprocessing needs neither)
# --------------------------------------------------------------------------- #
def _get(url, requests):
    return requests.get(url, impersonate="chrome110", timeout=20)


def get_province_links(requests, BeautifulSoup):
    html = _get(f"{BASE_URL}/tra-cuu-ma-so-thue-theo-tinh", requests)
    soup = BeautifulSoup(html.text, "lxml")
    links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/tra-cuu-ma-so-thue-theo-tinh/" in href:
            links[a.get_text(strip=True)] = BASE_URL + href
    return links


def _addresses_from_page(url, requests, BeautifulSoup):
    try:
        html = _get(url, requests)
    except Exception as e:
        print("  error:", e)
        return []
    soup = BeautifulSoup(html.text, "lxml")
    out = []
    for tag in soup.find_all("address"):
        addr = " ".join(tag.get_text(" ", strip=True).split())
        if len(addr) > 15:
            out.append(addr)
    return out


def crawl_province(province_url, target_size, requests, BeautifulSoup, pause=1.0):
    collected, page = [], 1
    while len(collected) < target_size:
        url = province_url if page == 1 else f"{province_url}?page={page}"
        addrs = _addresses_from_page(url, requests, BeautifulSoup)
        if not addrs:
            break
        collected.extend(addrs)
        page += 1
        time.sleep(pause)
    # de-dup, keep order, cap at target
    return list(dict.fromkeys(collected))[:target_size]


def crawl_all(target_size, out_path, pause=1.0):
    from curl_cffi import requests            # noqa: E402  (lazy)
    from bs4 import BeautifulSoup             # noqa: E402
    import pandas as pd

    provinces = get_province_links(requests, BeautifulSoup)
    print(f"[crawl] {len(provinces)} provinces; {target_size} addresses each")
    rows = []
    for name, url in provinces.items():
        addrs = crawl_province(url, target_size, requests, BeautifulSoup, pause)
        print(f"  {name:28} {len(addrs):3} addresses")
        rows.extend(addrs)
    s = pd.Series(list(dict.fromkeys(rows)), name="address")
    C.save_series(s, out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Preprocess
# --------------------------------------------------------------------------- #
def _drop_last_segment(text: str) -> str:
    parts = str(text).split(",")
    return ", ".join(parts[:-1]).strip() if len(parts) > 1 else str(text).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(C.RAW_DIR,
                    "addresses_50_per_city.csv"))
    ap.add_argument("--column", default="address")
    ap.add_argument("--output", default=os.path.join(C.PROC_DIR, "masothue_final.csv"))
    ap.add_argument("--crawl", action="store_true",
                    help="scrape masothue.com into --input first")
    ap.add_argument("--per-province", type=int, default=50)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between pages")
    ap.add_argument("--no-strip-country", action="store_true",
                    help="keep the trailing ', Việt Nam' segment")
    ap.add_argument("--no-segment", action="store_true")
    ap.add_argument("--no-eda", action="store_true")
    args = ap.parse_args()

    if args.crawl:
        crawl_all(args.per_province, args.input, args.pause)

    print(f"[Dataset 4 · masothue] {args.input}")
    s = C.read_column(args.input, args.column)
    if not args.no_strip_country:
        s = s.apply(_drop_last_segment)
    s = C.normalize_series(s)
    s = C.deduplicate(s, "masothue")
    C.finalize(s, args.output, segment=not args.no_segment,
               eda=not args.no_eda, name="data4_masothue", eda_dir=C.EDA_DIR)


if __name__ == "__main__":
    main()
