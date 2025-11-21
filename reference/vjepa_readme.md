Skip to content
Navigation Menu
facebookresearch
vjepa2

Type / to search
Code
Issues
26
Pull requests
5
Actions
Security
Insights
Owner avatar
vjepa2
Public
facebookresearch/vjepa2
Go to file
t
Name		
mmuckley
mmuckley
Add decord instructions to README (#82)
c2963a4
 · 
3 months ago
.github/workflows
Initial commit
5 months ago
app
fixing typos in 3 files (#78)
3 months ago
assets
Initial commit
5 months ago
configs
Comments addressing PR #15 and Issue #65 (#75)
3 months ago
evals
fixing typos in 3 files (#78)
3 months ago
notebooks
fixing typos in various texts
5 months ago
src
fixing typos in 3 files (#78)
3 months ago
tests
fixing miscellaneous typos in different files (#77)
3 months ago
.flake8
Initial commit
5 months ago
.gitignore
Initial commit
5 months ago
APACHE-LICENSE
Initial commit
5 months ago
CHANGELOG.md
Initial commit
5 months ago
CODE_OF_CONDUCT.md
Initial commit
5 months ago
CONTRIBUTING.md
Initial commit
5 months ago
LICENSE
Initial commit
5 months ago
README.md
Add decord instructions to README (#82)
3 months ago
hubconf.py
Initial commit
5 months ago
pyproject.toml
Initial commit
5 months ago
requirements-test.txt
Initial commit
5 months ago
requirements.txt
Initial commit
5 months ago
setup.py
Initial commit
5 months ago
Repository files navigation
README
Code of conduct
Contributing
MIT license
Apache-2.0 license
Security
V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
Meta FAIR
Mahmoud Assran∗, Adrien Bardes∗, David Fan∗, Quentin Garrido∗, Russell Howes∗, Mojtaba Komeili∗, Matthew Muckley∗, Ammar Rizvi∗, Claire Roberts∗, Koustuv Sinha∗, Artem Zholus*, Sergio Arnaud*, Abha Gejji*, Ada Martin*, Francois Robert Hogan*, Daniel Dugas*, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier*, Yann LeCun*, Michael Rabbat*, Nicolas Ballas*

*Core Team

[Paper] [Blog] [BibTex]

Official Pytorch codebase for V-JEPA 2 and V-JEPA 2-AC.

V-JEPA 2 is a self-supervised approach to training video encoders, using internet-scale video data, that attains state-of-the-art performance on motion understanding and human action anticipation tasks. V-JEPA 2-AC is a latent action-conditioned world model post-trained from V-JEPA 2 (using a small amount of robot trajectory interaction data) that solves robot manipulation tasks without environment-specific data collection or task-specific training or calibration.



V-JEPA 2 Pre-training
(Top) The encoder and predictor are pre-trained through self-supervised learning from video using a masked latent feature prediction objective, leveraging abundant natural videos to bootstrap physical world understanding and prediction. (Bottom) Performance of V-JEPA 2 on downstream understanding and prediction tasks.

 

Benchmark	VJEPA 2	Previous Best
EK100	39.7%	27.6% (PlausiVL)
SSv2 (Probe)	77.3%	69.7% (InternVideo2-1B)
Diving48 (Probe)	90.2%	86.4% (InternVideo2-1B)
MVP (Video QA)	44.5%	39.9% (InternVL-2.5)
TempCompass (Video QA)	76.9%	75.3% (Tarsier 2)
V-JEPA 2-AC Post-training
(Top) After post-training with a small amount of robot data, we can deploy the model on a robot arm in new environments, and tackle foundational tasks like reaching, grasping, and pick-and-place by planning from image goals. (Bottom) Performance on robot manipulation tasks using a Franka arm, with input provided through a monocular RGB camera.

 

Grasp	Pick-and-Place
Method	Reach	Cup	Box	Cup	Box
Octo	100%	10%	0%	10%	10%
Cosmos	80%	0%	20%	0%	0%
VJEPA 2-AC	100%	60%	20%	80%	50%
Models
V-JEPA 2
HuggingFace
See our HuggingFace collection for V-JEPA 2.

Pretrained Checkpoints
Model	#Parameters	Resolution	Download Link	Pretraining Config
ViT-L/16	300M	256	checkpoint	configs
ViT-H/16	600M	256	checkpoint	configs
ViT-g/16	1B	256	checkpoint	configs
ViT-g/16384	1B	384	checkpoint	configs
Pretrained backbones (via PyTorch Hub)
Please install Pytorch, timm and einops locally, then run the following to load each model. Installing Pytorch with CUDA support is strongly recommended.

import torch

# preprocessor
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
# models
vjepa2_vit_large = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
vjepa2_vit_huge = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
vjepa2_vit_giant = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
vjepa2_vit_giant_384 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant_384')
Pretrained checkpoints on Huggingface
You can also use our pretrained checkpoints on Huggingface.

from transformers import AutoVideoProcessor, AutoModel

hf_repo = "facebook/vjepa2-vitg-fpc64-256"
# facebook/vjepa2-vitl-fpc64-256
# facebook/vjepa2-vith-fpc64-256
# facebook/vjepa2-vitg-fpc64-256
# facebook/vjepa2-vitg-fpc64-384


model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)
Evaluation Attentive Probes
We share the trained attentive probes for two of our visual understanding evals (Something-Something v2 and Diving48) and the action anticipation eval EPIC-KITCHENS-100.

Model	SSv2	Diving48	EK100
Checkpoint	Training Config	Inference Config	Result	Checkpoint	Training Config	Inference Config	Result	Checkpoint	Training Config	Inference Config	Result
ViT-L/16	checkpoint	config	config	73.7%	checkpoint	config	config	89.0%	checkpoint	config	config	32.7 R@5
ViT-g/16384	checkpoint	config	config	77.3%	checkpoint	config	config	90.2%	checkpoint	config	config	39.7 R@5
V-JEPA 2-AC
Our action-conditioned checkpoint was trained from the ViT-g encoder.

Model	Download Link	Training Config
ViT-g/16	checkpoint	config
Pretrained action-conditioned backbone (via PyTorch Hub)
Please install Pytorch, timm and einops locally, then run the following to load each model. Installing Pytorch with CUDA support is strongly recommended.

import torch

vjepa2_encoder, vjepa2_ac_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
See energy_landscape_example.ipynb for an example notebook computing the energy landscape of the pretrained action-conditioned backbone using a robot trajectory collected from our lab. To run this notebook, you'll need to additionally install Jupyter and Scipy in your conda environment.

Getting Started
Setup
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .  # or `pip install -e .` for development mode
Note to macOS users: V-JEPA 2 relies on decord, which does not support macOS (and, unfortunately, is also no longer under development). In order to run the V-JEPA 2 code on macOS, you will need a different decord implementation. We do not make specific recommendations, although some users have reported the use of eva-decord (see PR 1) or decord2 (see PR 31). We leave the selection of the decord package up to the user's discretion.

Usage Demo
See vjepa2_demo.ipynb (Colab Link) or vjepa2_demo.py for an example of how to load both the HuggingFace and PyTorch V-JEPA 2 models and run inference on a sample video to get a sample classification result.

The script assumes the presence of downloaded model checkpoints so you will need to download the model weights and update the corresponding paths in the script. E.g.:

wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -P YOUR_DIR
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt -P YOUR_DIR

# Then update your model paths in vjepa2_demo.py.
pt_model_path = YOUR_DIR/vitg-384.pt
classifier_model_path = YOUR_DIR/ssv2-vitg-384-64x2x3.pt

# Then run the script (assumes your machine has a GPU)
python -m notebooks.vjepa2_demo
Probe-based evaluation
Probe-based evaluation consists in training an attentive probe on top of frozen V-JEPA 2 features. We provide training scripts for training your own probes, and checkpoints to run inference directly.

Training probes
Evaluations can be run either locally, or distributed via SLURM. (Running locally is useful for debugging and validation). These sample commands launch Something-Something v2 video classification; other evals are launched by specifying the corresponding config. Use provided training configs under "Evaluation Attentive Probes". These configs allow to train multiple probes in parallel with various optimization parameters. Change filepaths as needed (e.g. folder, checkpoint, dataset_train, dataset_val) to match locations of data and downloaded checkpoints on your local filesystem. Change # nodes and local batch size as needed to not exceed available GPU memory.

Local
To run locally, specify the GPUs to use on

python -m evals.main --fname configs/eval/vitl16/ssv2.yaml \
  --devices cuda:0 cuda:1
Distributed
python -m evals.main_distributed \
  --fname configs/eval/vitl/ssv2.yaml  \
  --time 8600 \
  --account my_account --qos=my_qos
Inference from existing probes
Use provided inference configs under Evaluation Attentive Probes. Download the corresponding checkpoint, rename it to 'latest.pt', and create a folder with the checkpoint inside, with the format matching the variables in the config:

[folder]/[eval_name]/[tag]/latest.pt
Then run inference, locally or distributed, using the same evaluation commands as above, but with configs from configs/inference.

Pretraining
Likewise, training can also be run locally or distributed. Pretraining and cooldown training phases are run with the same command using different configs. These sample commands launch initial training of a ViT-L model. Configs for cooldown (or action-conditioned) training can be found in the same directory as the config for initial training.

Local
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0
Distributed
python -m app.main_distributed \
  --fname configs/train/vitl16/pretrain-256px-16f.yaml
  --time 6000
  --account my_account --qos=my_qos
Postraining
Post-training of the action-conditioned model, starting from the pretrained VJEPA 2 backbone, also follows a similar interface, and can be run locally or distributed using this config. We post-train the model starting from the ViT-g/16 backbone.

Local
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0
Distributed
python -m app.main_distributed \
  --fname configs/train/vitg16/droid-256px-8f.yaml
  --time 6000
  --account my_account --qos=my_qos
Code Structure
.
├── app                              # training loops
│   ├── vjepa                        #   video JEPA pre-training
│   ├── vjepa_droid                  #   training the action-conditioned model
│   ├── main_distributed.py          #   entrypoint for launch app on slurm cluster
│   └── main.py                      #   entrypoint for launch app locally on your machine
├── configs                          # config files with experiment params for training and evaluation
│   ├── train                        #   pretraining (phase 1), cooldown (phase 2), and action-conditioned training
│   └── eval                         #   frozen evaluations
├── evals                            # evaluation loops training an attentive probe with frozen backbone...
│   ├── action_anticipation_frozen   #   action anticipation
│   ├── image_classification_frozen  #   image understanding
│   ├── video_classification_frozen  #   video understanding
│   ├── main_distributed.py          #   entrypoint for distributed evaluations
│   └── main.py                      #   entrypoint for locally-run evaluations
├── src                              # the package
│   ├── datasets                     #   datasets, data loaders, ...
│   ├── models                       #   model definitions
│   ├── masks                        #   mask collators, masking utilities, ...
│   └── utils                        #   shared utilities
├── tests                            # unit tests for some modules in `src`

License
The majority of V-JEPA 2 is licensed under MIT, however portions of the project are available under separate license terms:

src/datasets/utils/video/randaugment.py
src/datasets/utils/video/randerase.py
src/datasets/utils/worker_init_fn.py

are licensed under the Apache 2.0 license.

Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation

@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
About
PyTorch code and models for VJEPA2 self-supervised learning from video.

Resources
 Readme
License
 MIT, Apache-2.0 licenses found
Code of conduct
 Code of conduct
Contributing
 Contributing
Security policy
 Security policy
 Activity
 Custom properties
Stars
 2.5k stars
Watchers
 30 watching
Forks
 238 forks
Report repository
Releases
No releases published
Packages
No packages published
Contributors
15
@mmuckley
@artemZholus
@koustuvsinha
@russellhowes
@MidoAssran
@dfan
@Adrien987k
@garridoq
@ballasnicolas
@didier-durand
@mikerabbat
@n3puiol
@kaankarakose
@stonesstones
@Basile-Terv
Languages
Python
95.5%
 
Jupyter Notebook
4.5%
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information

