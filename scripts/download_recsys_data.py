import os
import sys
import urllib.request
import ssl

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_recsys")
os.makedirs(DATA_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def download_file(urls, target_path, name):
    if not isinstance(urls, list):
        urls = [urls]
    
    print(f"[*] Downloading {name}...")
    success = False
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            size_kb = os.path.getsize(target_path) / 1024
            if size_kb > 0:
                print(f"[OK] {name} downloaded -> {os.path.basename(target_path)} ({size_kb:.2f} KB)")
                success = True
                break
        except Exception as e:
            print(f"[!] Failed from {url}: {e}")
    return success

def main():
    print("=== Downloading Real-World MovieLens Interaction & Catalog Dataset ===")
    
    ratings_urls = [
        "https://raw.githubusercontent.com/khanhnamle1994/movielens/master/ratings.csv",
        "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/ml-1m.train.rating"
    ]
    movies_urls = [
        "https://raw.githubusercontent.com/khanhnamle1994/movielens/master/movies.csv"
    ]
    users_urls = [
        "https://raw.githubusercontent.com/khanhnamle1994/movielens/master/users.csv"
    ]

    r_ok = download_file(ratings_urls, os.path.join(DATA_DIR, "ratings.csv"), "MovieLens Ratings Logs")
    m_ok = download_file(movies_urls, os.path.join(DATA_DIR, "movies.csv"), "Movie Catalog Metadata")
    u_ok = download_file(users_urls, os.path.join(DATA_DIR, "users.csv"), "User Demographics Data")

    print("\n=== Dataset Summary in ./data_recsys/ ===")
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f" - {fname} ({size_kb:.2f} KB)")

if __name__ == "__main__":
    main()
