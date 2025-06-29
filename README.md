# 🎬 Computer Vision — Individual Assignment  
_Effects & Object Manipulation with OpenCV_

This project processes a one-minute clip from **“tarantino1.mp4”** and demonstrates a toolbox of classic computer-vision techniques:

* grayscale ⇄ colour switching  
* Gaussian / bilateral filtering  
* RGB & HSV segmentation with morphology  
* adaptive Sobel + Hough-circle detection  
* template matching & likelihood mapping  
* three “carte-blanche” effects:  
  1. colour-cycling object  
  2. “poof” disappearance into smoke  
  3. live replacement of a ball with a pizza 🍕

The pipeline runs in real-time on a laptop CPU.

You can see the final results in **“tarantino1.mp4”**!

---

## 📂 Project layout

```text
cv-video-editing/
├── data/
│   ├── tarantino1.mp4          # input video (tracked via Git LFS)
│   ├── background.png          # static frame of the empty scene
│   ├── ball_template.jpg       # template extracted by automatic_red_ball_detection.py
│   ├── smoke.png               # PNG with alpha channel
│   └── pizza.png               # PNG with alpha channel
├── src/
│   ├── Individual.py           # main processing pipeline (entry point)
│   └── automatic_red_ball_detection.py  # helper to auto-crop the ball template
├── docs/
│   └── demo.gif                # 6-sec showcase (optional)
├── requirements.txt
├── .gitattributes              # Git LFS config (tracks *.mp4)
└── README.md
```

---

## 🚀 Quick-start

```bash
# Clone & enter
git clone https://github.com/georgeforoglou/cv-video-editing.git
cd cv-video-editing

# 1) One-time Git LFS setup (skip if already installed)
#    macOS:   brew install git-lfs
#    Ubuntu:  sudo apt-get install git-lfs
git lfs install
git lfs pull          # fetches tarantino1.mp4

# 2) Create and activate a virtual env
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) (Optional) Re-crop the ball template
python src/automatic_red_ball_detection.py   # writes data/ball_template.jpg

# 4) Run the pipeline – output is saved as output.mp4
python src/Individual.py

python src/automatic_red_ball_detection.py            # writes data/ball_template.jpg

2) Run the main pipeline — output saved as output.mp4
python src/Individual.py
```

---

## 📝 License
Released under the MIT License.
© 2025 Georgios Foroglou · KU Leuven Computer Vision course.
