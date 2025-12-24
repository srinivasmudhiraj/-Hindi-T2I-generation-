Hindi T2I Generation

## 📦 Requirements

Make sure the following dependencies are installed:

```bash
tensorflow
tensorflow-gpu
transformers
farasapy
python-bidi
reshaper
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

References
------------------------------------------------------
Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. Caltech-UCSD Birds-200-2011 Dataset. The Caltech Institute Archives (California Institute of Technology). https://doi.org/10.57702/kvkwb8bo
