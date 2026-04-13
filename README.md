# 🍅 Tomato AI Robot (Detection + Segmentation + RL + 3D Picking)

A full end-to-end AI + Robotics project for **automatic tomato detection, segmentation, 3D localization, and smart harvesting order planning using Reinforcement Learning**.

---

## 🚀 Features

- 🍅 Tomato detection (YOLOv8)
- 🎯 Instance segmentation (YOLOv8-seg)
- 📷 RGB + Depth 3D position estimation
- 🧠 Cluster-based grouping
- 🤖 Reinforcement Learning (picking order)
- 🦾 Robot-ready pipeline (ROS-compatible design)

---

## 📁 Project Structure

```bash
tomato_ai_robot/
│
├── data/
│   ├── images/        # input images
│   ├── labels/        # YOLO labels
│   ├── masks/         # segmentation masks
│
├── models/
│
├── src/
│   ├── train_detection.py
│   ├── train_segmentation.py
│   ├── segment.py
│   ├── cluster.py
│   ├── features.py
│   ├── depth.py
│   ├── rl_env.py
│   ├── rl_train.py
│   ├── pipeline.py
│
└── main.py

## 🛠️ Recommended Conda Environment

```bash
conda create -n tomato python=3.10
conda activate tomato

# Install pip packages from requirements.txt
pip install -r requirements.txt

## full pipeline
cd tomato_ai_robot
python main.py

# ## run web app
# python app.py
```
