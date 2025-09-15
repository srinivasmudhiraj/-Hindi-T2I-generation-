Hindi T2I Generation

## 📦 Requirements

Make sure the following dependencies are installed:

```bash
tensorflow
tensorflow-gpu
transformers
PyArabic
farasapy
python-bidi
arabic-reshaper
tensorboard
tqdm

📂 Data Preparation
------------------------------------------------------
1. Download the CUB dataset from: https://www.vision.caltech.edu/datasets/
2. Preprocess and organize the dataset as required by the configuration files under cfg/.

🚀 Training
------------------------------------------------------
To train the GAN model on the Hindi captions + images:

python main_train.py --cfg cfg/bird.yml


📌 Notes
------------------------------------------------------
- Ensure GPU support (tensorflow-gpu) is installed for faster training.
- The project uses Hindi captions mapped with the CUB dataset.
