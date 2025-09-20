# txt_cluster_service.py
import os
import time
import json
import threading
import logging
from pathlib import Path
from queue import Queue, Empty

# Monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ML
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import sqlite3

# Optional OpenAI embeddings
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# Windows service
try:
    import servicemanager
    import win32serviceutil
    import win32service
    import win32event
    HAS_PYWIN32 = True
except Exception:
    HAS_PYWIN32 = False

from dotenv import load_dotenv
load_dotenv()

# ---------- Config ----------
INCOMING_DIR = Path(os.getenv("INCOMING_DIR", "C:/watched/incoming"))
CLUSTERS_DIR = Path(os.getenv("CLUSTERS_DIR", "C:/watched/clusters"))
DB_PATH = Path(os.getenv("DB_PATH", "C:/watched/metadata.db"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "C:/watched/model"))
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("1","true","yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MIN_DOC_LENGTH = 30  # characters
RECLUSTER_AFTER = 50  # yeni dosyadan sonra yeniden küme oluştur
LOGFILE = Path(os.getenv("LOGFILE", "C:/watched/service.log"))

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(CLUSTERS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Logging
logging.basicConfig(filename=str(LOGFILE), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("txt-cluster")

if USE_OPENAI and HAS_OPENAI:
    openai.api_key = OPENAI_API_KEY

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE,
        filename TEXT,
        cluster INTEGER,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        doc_id INTEGER PRIMARY KEY,
        vector BLOB,
        FOREIGN KEY(doc_id) REFERENCES docs(id)
    )""")
    conn.commit()
    conn.close()

def add_doc_record(path, filename, cluster=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO docs(path, filename, cluster) VALUES(?,?,?)", (str(path), filename, cluster))
    conn.commit()
    conn.close()

def update_doc_cluster(path, cluster):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE docs SET cluster=? WHERE path=?", (int(cluster), str(path)))
    conn.commit()
    conn.close()

# ---------- Embedding helpers ----------
def embed_texts_openai(texts):
    # example using OpenAI text-embedding-3-small or similar
    results = []
    for t in texts:
        if not t.strip():
            results.append(np.zeros(1536))
            continue
        resp = openai.Embedding.create(input=t, model="text-embedding-3-small")
        vec = np.array(resp["data"][0]["embedding"], dtype=np.float32)
        results.append(vec)
    return np.vstack(results)

def embed_texts_tfidf(texts):
    vect_path = MODEL_DIR / "tfidf.joblib"
    svd_path = MODEL_DIR / "svd.joblib"
    if vect_path.exists() and svd_path.exists():
        tfidf = joblib.load(vect_path)
        svd = joblib.load(svd_path)
    else:
        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
        X = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=128, random_state=42)
        joblib.dump(tfidf, vect_path)
        joblib.dump(svd, svd_path)
        svd.fit(X)
    X = tfidf.transform(texts)
    Xr = svd.transform(X)
    return Xr

def embed_texts(texts):
    if USE_OPENAI and HAS_OPENAI:
        try:
            return embed_texts_openai(texts)
        except Exception as e:
            logger.exception("OpenAI embedding failed, falling back to TF-IDF: %s", e)
    return embed_texts_tfidf(texts)

# ---------- Clustering ----------
def choose_k_and_cluster(X, k_max=10):
    # Basit: dene k=2..k_max, silhouette'a bak
    best_k = 2
    best_score = -1
    best_labels = None
    for k in range(2, min(k_max, max(2, X.shape[0]))+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        logger.info("k=%d silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = km
    return best_k, best_labels, best_model

def perform_recluster(all_texts, all_paths):
    if not all_texts:
        return {}
    X = embed_texts(all_texts)
    k, labels, model = choose_k_and_cluster(X, k_max=10)
    mapping = {}
    for p, l in zip(all_paths, labels):
        mapping[p] = int(l)
    # persist model if desired
    joblib.dump(model, MODEL_DIR / "kmeans.joblib")
    logger.info("Reclustered: k=%d", k)
    return mapping

# ---------- File watcher ----------
class TxtHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".txt"):
            logger.info("Created: %s", event.src_path)
            self.queue.put(event.src_path)
    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".txt"):
            logger.info("Modified: %s", event.src_path)
            self.queue.put(event.src_path)

# ---------- Worker ----------
class Processor(threading.Thread):
    def __init__(self, queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.seen = set()
        self.counter = 0

    def run(self):
        while True:
            try:
                path = self.queue.get(timeout=1)
            except Empty:
                continue
            try:
                self.process_file(Path(path))
            except Exception as e:
                logger.exception("Processing failed for %s: %s", path, e)

    def process_file(self, path: Path):
        # read file
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = path.read_text(encoding="latin-1", errors="ignore")
        if len(text.strip()) < MIN_DOC_LENGTH:
            logger.info("Skipping short file: %s", path)
            return
        add_doc_record(path, path.name)
        # increment counter and maybe retrain/recluster
        self.counter += 1
        if self.counter >= RECLUSTER_AFTER:
            self.counter = 0
            self.recluster_all()
        else:
            # simple incremental: compute embedding and assign to nearest cluster if possible
            self.incremental_assign(path, text)

    def incremental_assign(self, path, text):
        # If we have a saved kmeans model, predict; otherwise defer to full recluster
        km_path = MODEL_DIR / "kmeans.joblib"
        if km_path.exists():
            model = joblib.load(km_path)
            vec = embed_texts([text])
            label = int(model.predict(vec)[0])
            target = CLUSTERS_DIR / f"cluster_{label}"
            target.mkdir(parents=True, exist_ok=True)
            dest = target / path.name
            path.replace(dest)
            update_doc_cluster(path, label)
            logger.info("Moved %s to %s", path, dest)
        else:
            logger.info("No kmeans model found, triggering full recluster")
            self.recluster_all()

    def recluster_all(self):
        # Load all docs in incoming + existing cluster dirs
        all_paths = []
        all_texts = []
        for p in INCOMING_DIR.rglob("*.txt"):
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                t = p.read_text(encoding="latin-1", errors="ignore")
            if len(t.strip()) < MIN_DOC_LENGTH:
                continue
            all_paths.append(str(p))
            all_texts.append(t)
        # Also include files currently in cluster folders
        for cdir in CLUSTERS_DIR.iterdir():
            if cdir.is_dir():
                for p in cdir.glob("*.txt"):
                    try:
                        t = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        t = p.read_text(encoding="latin-1", errors="ignore")
                    if len(t.strip()) < MIN_DOC_LENGTH:
                        continue
                    all_paths.append(str(p))
                    all_texts.append(t)

        mapping = perform_recluster(all_texts, all_paths)
        # move files
        for pstr, cluster in mapping.items():
            p = Path(pstr)
            target = CLUSTERS_DIR / f"cluster_{cluster}"
            target.mkdir(parents=True, exist_ok=True)
            try:
                dest = target / p.name
                if p.exists():
                    p.replace(dest)
                update_doc_cluster(p, cluster)
            except Exception:
                logger.exception("Failed to move %s -> cluster_%d", p, cluster)
        logger.info("Recluster all finished, moved %d files", len(mapping))

# ---------- Service wrapper (optional) ----------
class TxtClusterService(win32serviceutil.ServiceFramework if HAS_PYWIN32 else object):
    if HAS_PYWIN32:
        _svc_name_ = "TxtClusterService"
        _svc_display_name_ = "Text Clustering Service"
        _svc_description_ = "Watches a folder and clusters text files into folders."

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
            self.stop_requested = False

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.hWaitStop)
            self.stop_requested = True

        def SvcDoRun(self):
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                          servicemanager.PYS_SERVICE_STARTED,
                          (self._svc_name_, ""))
            run_main_loop()


# ---------- Main loop ----------
def run_main_loop():
    init_db()
    q = Queue()
    event_handler = TxtHandler(q)
    observer = Observer()
    observer.schedule(event_handler, str(INCOMING_DIR), recursive=True)
    observer.start()
    worker = Processor(q)
    worker.start()
    logger.info("Started watching %s", INCOMING_DIR)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and HAS_PYWIN32 and sys.argv[1] in ("install","remove","start","stop"):
        # pass through to pywin32 helper
        win32serviceutil.HandleCommandLine(TxtClusterService)
    else:
        # run as console
        logger.info("Starting in console mode")
        run_main_loop()
