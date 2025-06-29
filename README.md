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

## 🚀 Quick-start

git clone https://github.com/georgeforoglou/cv-video-editing.git
cd cv-video-editing

# If Git LFS is not installed, do it once:
#   sudo apt-get install git-lfs   # Debian/Ubuntu
#   brew install git-lfs           # macOS
git lfs pull                       # downloads the video

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Generate / update the ball template (optional)
python src/automatic_red_ball_detection.py            # writes data/ball_template.jpg

# 2) Run the main pipeline — output saved as output.mp4
python src/Individual.py
