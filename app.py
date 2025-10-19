import os
import io
import zipfile
import pickle
import tempfile
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

from PIL import Image
import imagehash
import cv2
import streamlit as st
from send2trash import send2trash

# -----------------------
# Config
# -----------------------
class Config:
    UPLOADS_DIR = os.path.join(os.getcwd(), "streamlit_uploads")
    CACHE_FILE_DEFAULT = "cache.pkl"
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.jfif')
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".gif")
    FRAME_INTERVAL_SEC = 2
    HAMMING_THRESHOLD = 15
    MIN_MATCHES = 7
    SAVE_INTERVAL_DEFAULT = 50
    WORKERS_DEFAULT = 8

    THRESHOLD_MODES = {
        "0–5(Same)": (0, 5),
        "6–10(litl Dif)": (6, 10),
        "11–20(Dif)": (11, 20)
    }

    _CPU_COUNT = os.cpu_count() or 1
    MAX_WORKERS = max(_CPU_COUNT - 1, 1)
    WORKERS_DEFAULT = min(WORKERS_DEFAULT, MAX_WORKERS)

os.makedirs(Config.UPLOADS_DIR, exist_ok=True)

# -----------------------
# Helpers functions
# -----------------------
def pick_folder_via_subprocess():
    script = r"""
import tkinter as tk
from tkinter import filedialog
import os, sys

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()
initial = os.path.expanduser("~")
folder = filedialog.askdirectory(parent=root, initialdir=initial)
try:
    root.destroy()
except Exception:
    pass
if not folder:
    folder = ""
print(folder)
"""
    tf = None
    try:
        tf = tempfile.NamedTemporaryFile('w', delete=False, suffix='.py', encoding='utf-8')
        tf.write(script)
        tf.close()
        res = subprocess.run([sys.executable, tf.name], capture_output=True, text=True, timeout=300)
        out = res.stdout.strip()
        return out
    except Exception:
        return ""
    finally:
        try:
            if tf:
                os.unlink(tf.name)
        except Exception:
            pass

def safe_delete(file_path):
    try:
        file_path = os.path.normpath(file_path)
        send2trash(file_path)
        return True, "Sent to OS Recycle Bin"
    except Exception as e:
        return False, str(e)

def compute_image_hash_worker(args):
    image_path, base_dir = args
    rel_path = os.path.relpath(image_path, base_dir)
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            quads = ImageSplitter.split_into_quadrants(img)
            hashes = []
            for q in quads:
                h = imagehash.phash(q)
                hashes.append(int(str(h), 16))
            return rel_path, hashes
    except Exception:
        return rel_path, None

def get_video_frame_hashes_worker(video_path_and_name):
    video_path, fname = video_path_and_name
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return fname, set()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = int(fps * Config.FRAME_INTERVAL_SEC) if fps > 0 else int(25 * Config.FRAME_INTERVAL_SEC)
        frame_hashes = set()
        frame_idx = 0
        success, frame = cap.read()
        while success:
            if frame_idx % max(1, frame_skip) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                h = imagehash.phash(pil_img)
                frame_hashes.add(int(str(h), 16))
            frame_idx += 1
            success, frame = cap.read()
        cap.release()
        return fname, frame_hashes
    except Exception:
        return fname, set()

# -----------------------
# Cache utils (class)
# -----------------------
class CacheManager:
    def __init__(self, directory, cache_filename):
        self.directory = directory
        self.cache_file = os.path.join(directory, cache_filename)

    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def save(self, cache):
        path = os.path.normpath(self.cache_file)
        with open(path, 'wb') as f:
            pickle.dump(cache, f)


# -----------------------
# Image splitting helper (4 quadrants)
# -----------------------
class ImageSplitter:
    @staticmethod
    def split_into_quadrants(pil_img):
        w, h = pil_img.size
        mid_x = w // 2
        mid_y = h // 2
        return [
            pil_img.crop((0, 0, mid_x, mid_y)),           # top-left
            pil_img.crop((mid_x, 0, w, mid_y)),           # top-right
            pil_img.crop((0, mid_y, mid_x, h)),           # bottom-left
            pil_img.crop((mid_x, mid_y, w, h)),           # bottom-right
        ]


# -----------------------
# ImageHasher (OOP)
# -----------------------
class ImageHasher:
    def __init__(self, base_dir, cache_filename, workers=1, save_interval=Config.SAVE_INTERVAL_DEFAULT):
        self.base_dir = base_dir
        self.cache = CacheManager(base_dir, cache_filename)
        requested = workers or Config.WORKERS_DEFAULT
        requested = max(1, int(requested))
        self.workers = min(requested, Config.MAX_WORKERS)
        self.save_interval = save_interval

    def build_hashes(self):
        cache = self.cache.load()
        all_files = [os.path.join(root, f)
                     for root, _, files in os.walk(self.base_dir)
                     for f in files if f.lower().endswith(Config.IMAGE_EXTENSIONS)]

        # remove missing entries from cache
        for fname in list(cache.keys()):
            if not os.path.exists(os.path.join(self.base_dir, fname)):
                del cache[fname]

        to_process = []
        for path in all_files:
            rel = os.path.relpath(path, self.base_dir)
            need = False
            if rel not in cache:
                need = True
            else:
                val = cache.get(rel)
                if not isinstance(val, list) or len(val) != 4:
                    try:
                        del cache[rel]
                    except Exception:
                        pass
                    need = True
            if need:
                to_process.append((path, self.base_dir))

        if not to_process:
            st.info("No new images to hash (cache up-to-date).")
            return cache

        progress_bar = st.progress(0)
        status_text = st.empty()
        i = 0
        total = len(to_process)

        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(compute_image_hash_worker, arg): arg for arg in to_process}
            for fut in as_completed(futures):
                try:
                    name, hlist = fut.result()
                    if hlist is not None and isinstance(hlist, list) and len(hlist) == 4:
                        cache[name] = hlist
                except Exception:
                    pass
                i += 1
                progress_bar.progress(i / total)
                status_text.text(f"Hashing images (quadrants): {i}/{total}")
                if i % self.save_interval == 0:
                    self.cache.save(cache)

        status_text.text(f"Finished hashing {i} images.")
        self.cache.save(cache)
        return cache


# -----------------------
# VideoHasher (OOP)
# -----------------------
class VideoHasher:
    def __init__(self, directory, cache_filename, workers=Config.MAX_WORKERS):
        self.directory = directory
        self.cache = CacheManager(directory, cache_filename)
        requested = workers or Config.WORKERS_DEFAULT
        requested = max(1, int(requested))
        self.workers = min(requested, Config.MAX_WORKERS)

    def build_hashes(self):
        cache = self.cache.load()
        all_files = [f for f in os.listdir(self.directory) if f.lower().endswith(Config.VIDEO_EXTENSIONS)]

        for saved_name in list(cache.keys()):
            if saved_name not in all_files:
                del cache[saved_name]

        to_process = [(os.path.join(self.directory, f), f) for f in all_files if f not in cache]

        if not to_process:
            st.info("No new videos to hash (cache up-to-date).")
            return cache

        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(to_process)
        save_counter = 0

        with Pool(processes=self.workers) as pool:
            for i, result_item in enumerate(pool.imap_unordered(get_video_frame_hashes_worker, to_process)):
                fname, hashes = result_item
                cache[fname] = hashes
                save_counter += 1
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Processing video {i + 1}/{total}: {fname}")
                if save_counter % Config.SAVE_INTERVAL_DEFAULT == 0:
                    self.cache.save(cache)

        status_text.text(f"Finished hashing {total} videos.")
        self.cache.save(cache)
        return cache


# -----------------------
# DuplicateFinder (OOP)
# -----------------------
class DuplicateFinder:
    @staticmethod
    def find_image_duplicates(hashes, threshold_range):
        names = list(hashes.keys())
        pairs = []
        visited = set()
        min_thr, max_thr = threshold_range

        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(names)

        for i in range(len(names)):
            ni = names[i]
            if ni in visited:
                progress_bar.progress((i + 1) / total)
                continue
            for j in range(i + 1, len(names)):
                nj = names[j]
                if nj in visited:
                    continue
                h_a = hashes.get(ni)
                h_b = hashes.get(nj)
                if not (isinstance(h_a, list) and isinstance(h_b, list) and len(h_a) == 4 and len(h_b) == 4):
                    continue
                d_sum = 0
                for k in range(4):
                    d_sum += (h_a[k] ^ h_b[k]).bit_count()
                avg_d = d_sum / 4.0
                if min_thr <= avg_d <= max_thr:
                    pairs.append([ni, nj])
                    visited.add(nj)
            visited.add(ni)
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Searching duplicates (quadrants): {i + 1}/{total}")

        status_text.text(f"Finished searching duplicates: {len(pairs)} pairs found.")
        return pairs

    @staticmethod
    def count_similar_hashes(hashes_a, hashes_b, threshold=Config.HAMMING_THRESHOLD):
        return sum(1 for ha in hashes_a for hb in hashes_b if (ha ^ hb).bit_count() <= threshold)

    @staticmethod
    def find_video_duplicates(video_hashes):
        filenames = list(video_hashes.keys())
        visited = set()
        duplicate_groups = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(filenames)

        for i in range(len(filenames)):
            fname_i = filenames[i]
            if fname_i in visited:
                progress_bar.progress((i + 1) / total)
                continue
            group = [(fname_i, 0)]
            hashes_i = video_hashes[fname_i]
            for j in range(i + 1, len(filenames)):
                fname_j = filenames[j]
                if fname_j in visited:
                    continue
                matches = DuplicateFinder.count_similar_hashes(hashes_i, video_hashes[fname_j])
                if matches >= Config.MIN_MATCHES:
                    group.append((fname_j, matches))
            if len(group) > 1:
                duplicate_groups.append(group)
                for f, _ in group:
                    visited.add(f)
            else:
                visited.add(fname_i)
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Searching video duplicates: {i + 1}/{total}")

        status_text.text(f"Finished searching video duplicates: {len(duplicate_groups)} groups found.")
        return duplicate_groups


# -----------------------
# UI class (Streamlit)
# -----------------------
class UI:
    def __init__(self):
        self.uploads_dir = Config.UPLOADS_DIR
        self.cache_default = Config.CACHE_FILE_DEFAULT

    def run(self):
        st.set_page_config(page_title="Duplicate Finder", layout="wide")
        st.title("🗂 Duplicate Finder — Images & Videos (OOP)")

        st.sidebar.header("Mode")
        mode = st.sidebar.radio("Select mode", ["Images", "Videos"])

        if 'last_mode' not in st.session_state:
            st.session_state.last_mode = mode
        if st.session_state.last_mode != mode:
            st.session_state.pairs = []
            st.session_state.idx = 0
            st.session_state.cache = {}
            st.session_state.last_mode = mode

        st.sidebar.header("Source")
        input_mode = st.sidebar.radio("Source type", ["📁 Local folder", "📤 Upload files", "📦 Upload ZIP"])

        if 'last_input_mode' not in st.session_state:
            st.session_state.last_input_mode = input_mode
        if st.session_state.last_input_mode != input_mode:
            st.session_state.pairs = []
            st.session_state.idx = 0
            st.session_state.cache = {}
            st.session_state.last_input_mode = input_mode

        st.sidebar.markdown("---")
        st.sidebar.header("Settings")

        cache_file = st.sidebar.text_input("Cache file", self.cache_default)

        workers = st.sidebar.number_input(
            "Workers",
            min_value=1,
            value=min(Config.WORKERS_DEFAULT, Config.MAX_WORKERS),
            max_value=Config.MAX_WORKERS
        )

        save_interval = st.sidebar.number_input("Save interval", min_value=1, value=Config.SAVE_INTERVAL_DEFAULT)
        threshold_mode = st.sidebar.radio("Hamming threshold mode", list(Config.THRESHOLD_MODES.keys()))
        threshold_range = Config.THRESHOLD_MODES[threshold_mode]

        if mode == "Videos":
            # update class attributes (no 'global' needed)
            Config.FRAME_INTERVAL_SEC = st.sidebar.slider("Frame interval (sec)", 1, 10, Config.FRAME_INTERVAL_SEC)
            Config.MIN_MATCHES = st.sidebar.slider("Minimum matching frames for duplicates", 1, 20, Config.MIN_MATCHES)

        base_dir = None
        uploaded_temp_dir = self.uploads_dir

        if input_mode.startswith("📁"):
            if 'selected_folder' not in st.session_state:
                st.session_state.selected_folder = ""
            if st.sidebar.button("📂 Select folder..."):
                picked = pick_folder_via_subprocess()
                if picked:
                    st.session_state.selected_folder = picked
            typed = st.sidebar.text_input("Or paste folder path", st.session_state.selected_folder or "")
            base_dir = st.session_state.selected_folder or typed
        elif input_mode.startswith("📤"):
            uploaded_files = st.sidebar.file_uploader("Upload files", type=None, accept_multiple_files=True)
            if uploaded_files:
                sub = os.path.join(uploaded_temp_dir, "uploaded_files")
                os.makedirs(sub, exist_ok=True)
                for uf in uploaded_files:
                    dest = os.path.join(sub, uf.name)
                    base, ext = os.path.splitext(dest)
                    k = 1
                    while os.path.exists(dest):
                        dest = f"{base}_{k}{ext}"
                        k += 1
                    with open(dest, "wb") as f:
                        f.write(uf.getbuffer())
                base_dir = sub
        elif input_mode.startswith("📦"):
            uploaded_zip = st.sidebar.file_uploader("Upload ZIP", type=['zip'])
            if uploaded_zip:
                sub = os.path.join(uploaded_temp_dir, "uploaded_zip")
                os.makedirs(sub, exist_ok=True)
                data = uploaded_zip.read()
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    z.extractall(sub)
                base_dir = sub

        if not base_dir:
            st.info("Select a source folder or upload files/ZIP to proceed.")
            return

        if 'cache' not in st.session_state:
            st.session_state.cache = {}
        if 'pairs' not in st.session_state:
            st.session_state.pairs = []
        if 'idx' not in st.session_state:
            st.session_state.idx = 0

        if st.sidebar.button("🔍 Scan / Update"):
            if mode == "Images":
                hasher = ImageHasher(base_dir, cache_file, workers=workers, save_interval=save_interval)
                st.session_state.cache = hasher.build_hashes()
                st.session_state.pairs = DuplicateFinder.find_image_duplicates(st.session_state.cache, threshold_range)
            else:
                vhasher = VideoHasher(base_dir, cache_file, workers=workers)
                st.session_state.cache = vhasher.build_hashes()
                st.session_state.pairs = DuplicateFinder.find_video_duplicates(st.session_state.cache)
            st.session_state.idx = 0
            st.success(f"Found {len(st.session_state.pairs)} duplicates.")

        if not st.session_state.pairs:
            st.info("No duplicates found.")
            return

        col_nav = st.columns([1, 1, 1, 4])
        if col_nav[0].button("<< First"):
            st.session_state.idx = 0
        if col_nav[1].button("< Prev"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
        if col_nav[2].button("Next >"):
            st.session_state.idx = min(len(st.session_state.pairs) - 1, st.session_state.idx + 1)
        col_nav[3].markdown(f"**Pair {st.session_state.idx + 1}/{len(st.session_state.pairs)}**")

        pair = st.session_state.pairs[st.session_state.idx]

        if mode == "Images":
            left, right = pair
            left_path = os.path.join(base_dir, left)
            right_path = os.path.join(base_dir, right)
            col1, col2 = st.columns(2)
            with col1:
                st.image(left_path, width='stretch')
                st.subheader(left)
                if st.button("Delete LEFT", key=f"del_left_{st.session_state.idx}"):
                    ok, msg = safe_delete(left_path)
                    if ok:
                        st.session_state.cache.pop(left, None)
                        st.session_state.pairs.pop(st.session_state.idx)
                        st.success("LEFT deleted.")
                    else:
                        st.error(msg)
            with col2:
                st.image(right_path, width='stretch')
                st.subheader(right)
                if st.button("Delete RIGHT", key=f"del_right_{st.session_state.idx}"):
                    ok, msg = safe_delete(right_path)
                    if ok:
                        st.session_state.cache.pop(right, None)
                        st.session_state.pairs.pop(st.session_state.idx)
                        st.success("RIGHT deleted.")
                    else:
                        st.error(msg)
        else:  # Videos
            col1, col2 = st.columns(2)
            for idx_vid, item in enumerate(pair[:2]):
                if isinstance(item, tuple):
                    fname, matches = item
                else:  # якщо просто ім'я файлу
                    fname, matches = item, 0
                video_path = os.path.join(base_dir, fname)
                if not os.path.exists(video_path):
                    st.warning(f"File {fname} not found — skipping.")
                    continue
                with (col1 if idx_vid == 0 else col2):
                    st.video(video_path)
                    st.subheader(f"{fname} — {matches} matching frames")
                    if st.button(f"Delete {fname}", key=f"del_vid_{st.session_state.idx}_{idx_vid}"):
                        ok, msg = safe_delete(video_path)
                        if ok:
                            st.session_state.cache.pop(fname, None)
                            st.session_state.pairs[st.session_state.idx].pop(idx_vid)
                            st.success(f"{fname} deleted.")
                        else:
                            st.error(msg)


# -----------------------
# Entry point
# -----------------------
def main():
    ui = UI()
    ui.run()


if __name__ == '__main__':
    main()
