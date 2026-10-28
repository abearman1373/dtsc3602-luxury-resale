"""Download every page of a Shopify store's public /products.json feed (allowed by its robots.txt)."""
import sys, os, json, time, urllib.request, urllib.error

base, out = sys.argv[1].rstrip("/"), sys.argv[2]
os.makedirs(out, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (DTSC 3602 student research project)"}

page, total = 1, 0
while True:
    path = os.path.join(out, f"page_{page:04d}.json")
    if not os.path.exists(path):
        for attempt in range(5):
            try:
                req = urllib.request.Request(f"{base}/products.json?limit=250&page={page}", headers=UA)
                data = urllib.request.urlopen(req, timeout=60).read()
                break
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"page {page}: {e}, retrying in {30 * (attempt + 1)}s", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit(f"gave up on page {page}")
        open(path, "wb").write(data)
        time.sleep(1.5)
    n = len(json.load(open(path, encoding="utf-8"))["products"])
    if n == 0:
        os.remove(path)
        break
    total += n
    if page % 10 == 0:
        print(f"page {page}: {total} products", flush=True)
    page += 1
print(f"done: {page - 1} pages, {total} products", flush=True)
