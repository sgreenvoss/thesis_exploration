import queue, threading, re
import cv2
import speech_recognition as sr
from faster_whisper import WhisperModel
from ultralytics import YOLOWorld
import os, sys

query_q = queue.Queue()
# hack to recognize the proper cuda lib stuff
if sys.platform == "win32":
    venv_root = os.path.dirname(sys.executable)
    nvidia_base = os.path.join(venv_root, "..", "Lib", "site-packages", "nvidia")
    dll_dirs = []
    for pkg in ("cublas", "cudnn"):
        bin_dir = os.path.join(nvidia_base, pkg, "bin")
        if os.path.isdir(bin_dir):
            dll_dirs.append(bin_dir)
    
    os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ["PATH"]

# ---------- speech side ----------
def extract_object(text: str):
    """Crude: grab whatever follows 'find/show/where is/look for (the|a)'.
    Swap for spaCy noun chunks or an LLM call if you want more robustness."""
    m = re.search(r"(?:find|show me|where is|where's|look for|detect)\s+(?:the|a|an|my)?\s*(.+)",
                  text.lower())
    if not m:
        return None
    return re.sub(r"[^\w\s]", "", m.group(1)).strip() or None

def listen_loop():
    whisper = WhisperModel("small", device="cuda", compute_type="float16")
    rec = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as mic:
        rec.adjust_for_ambient_noise(mic)
        while True:
            audio = rec.listen(mic, phrase_time_limit=5)
            
            import numpy as np
            pcm = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = whisper.transcribe(pcm, language="en")
            text = " ".join(s.text for s in segments).strip()
            if not text:
                continue
            print("heard:", text)
            obj = extract_object(text)
            if obj:
                query_q.put(obj)

# ---------- vision side ----------
def main():
    options = ["person", "glasses"]
    threading.Thread(target=listen_loop, daemon=True).start()

    model = YOLOWorld("yolov8s-worldv2.pt")   # downloads on first use
    model.set_classes(options)
    current = None

    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # pick up newest spoken query, if any
        try:
            while True:
                query = query_q.get_nowait()
                if query in options:
                #model.set_classes([current])
                    current = query
                    print("looking for", query)
                else:
                    print("I don't know", query)
        except queue.Empty:
            pass

        if current:
            idx = options.index(current)
            res = model.predict(frame, conf=0.25, verbose=False)[0]
            for box in res.boxes:
                if box.cls[0] == idx:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{current} {float(box.conf):.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"searching: {current}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("voice detect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()