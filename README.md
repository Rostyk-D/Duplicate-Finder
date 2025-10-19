# 🗂 Duplicate Finder — Images & Videos

**Duplicate Finder** is an application on **Streamlit** that automatically finds duplicate images and videos using **perceptual hashes (pHash)**.
The program supports multiprocessing, caching of results, and a convenient interface for viewing and removing duplicates.
<img width="1915" height="963" alt="image" src="https://github.com/user-attachments/assets/e5f6357a-ba1c-4cbf-a691-470c5474e196" />

---

## 🚀 Main feauters

✅ Search for duplicate **images** based on content similarity (not just names).  
✅ Search for duplicate **videos** based on frame similarity.  
✅ Multi-processor support for fast processing.  
✅ **Caching** of file hashes to save time during repeated checks.  
✅ Determination of **similarity degree** via a configurable threshold (Hamming distance).  
✅ Ability to delete duplicates in **Recycle Bin** (via `send2trash`).  
✅ Convenient Streamlit interface with **image/video preview**.  
✅ Support for selecting a local folder, uploading individual files or ZIP archives.

---

## 🧠 How this work

### 🔹 Image
1. Each image is divided into **4 quadrants**.
2. For each quadrant, a **pHash** (perceptual hash) is calculated.
3. All hashes are compared, and if the average **Hamming distance** between them is lower than a given threshold, the files are considered similar.

### 🔹 Video
1. Frames are extracted from the video every *N seconds* (`Config.FRAME_INTERVAL_SEC`).
2. **pHash** is calculated for each frame.
3. If two videos have enough similar frames (by the number of `Config.MIN_MATCHES`), they are considered duplicates.
---

## ⚙️ Technologies used

- 🐍 **Python 3.12+**
- 🖼 **Pillow** — working with images
- 🧮 **imagehash** — calculating perceptual hashes
- 🎥 **OpenCV (cv2)** — working with video
- 💾 **pickle** — caching hashes
- 🔄 **concurrent.futures / multiprocessing** — parallel processing
- 🧱 **Streamlit** — building interactive UI
- 🗑 **send2trash** — safe file deletion to the trash
- 🧰 **Tki

---

## 🧰 Startup Instructions

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

or manually:
```bash
pip install streamlit pillow imagehash opencv-python send2trash
```

---

### 2️⃣ Run the application
```bash
streamlit run app.py
```

---

### 3️⃣ Usage
1. Select the mode:
  - **Images** – to search for duplicate images
  - **Videos** – to search for duplicate videos
2. Specify the source:
  - 📁 **Local folder** — select a folder with files
  - 📤 **Upload files** — upload individual files
  - 📦 **Upload ZIP** — upload an archive with files
3. Click **🔍 Scan / Update** to create or update the hash cache.
4. View the found duplicates, delete unnecessary files (via the trash).
---

## ⚙️ Configuration

| Parameter | Description | Default value |
|-----------|------|--------------------------|
| `FRAME_INTERVAL_SEC` | Interval between frames when analyzing video | 2 sec |
| `HAMMING_THRESHOLD` | Frame similarity threshold | 15 |
| `MIN_MATCHES` | Minimum number of similar frames between videos | 7 |
| `SAVE_INTERVAL_DEFAULT` | How often to save the cache during processing | 50 |
| `WORKERS_DEFAULT` | Number of processes | 8 (or less than the number of CPUs) |

---

## 📁 Project structure

```
duplicate_finder/
│
├── app.py               # Main program file
├── streamlit_uploads/   # Temporary upload files
├── requirements.txt     # Dependency list
└── README.md            # Project description
```

---

## 🧩 Main Classes

| Class | Purpose |
|------|---------------|
| `Config` | Stores global application settings |
| `CacheManager` | Loads and stores hash cache |
| `ImageSplitter` | Splits images into quadrants |
| `ImageHasher` | Computes hashes for images |
| `VideoHasher` | Computes hashes for videos |
| `DuplicateFinder` | Compares hashes and finds duplicates |
| `UI` | Responsible

---

## 🧹 Secure deletion
Files are not permanently deleted - they are **moved to the trash** via the `send2trash` library.
