import queue, threading, re
import cv2
import speech_recognition as sr
from faster_whisper import WhisperModel
from ultralytics import YOLO
import os, sys

vid = "video/oregon_wildfire.mp4"
models = ["yolo_models/v26-fire-detection.pt", "yolo_models/fire-uav.pt", "yolo_models/v10-fire-and-smoke.pt"]

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


def main():
    file = open("csvs/all_models_oregon_vid.csv", "w")
    file.write("model,preprocess,inference,postprocess\n")
    cap = cv2.VideoCapture(vid)

    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('all_models.mp4', fourcc, fps, (f_width, f_height))
    #options = ["person", "glasses", "wildfire"]
    #threading.Thread(target=listen_loop, daemon=True).start()
    real_models = [YOLO(mod) for mod in models]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for i in range(len(real_models)):
            model = real_models[i]
            res = model.predict(frame, conf=0.25, verbose=False)[0]
            speed = res.speed
            file.write(f"{models[i]},{speed["preprocess"]},{speed["inference"]},{speed["postprocess"]}\n")
            for box in res.boxes:
                # if box.cls[0] == idx:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
                    cv2.putText(frame, f"{models[i][12:]}: {model.names[int(box.cls)]}", (x1, y1 - 8),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
            # cv2.putText(frame, f"searching: {current}", (10, 25),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("fire detect", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            file.close()
            break
    file.close()
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()