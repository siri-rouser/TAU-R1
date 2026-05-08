# [TAU-R1: Visual Language Model for Traffic Anomaly Understanding](https://arxiv.org/abs/2603.19098)

**TAU-R1(Traffic Anomaly Understanding - R1)** is a VLM-based two-layer hiearchichal framework aiming to solve traffic anomaly understanding task on real-world roundabout scenarios. 

## Dataset and Models
- **Dataset**: [Roundabout-TAU](https://huggingface.co/datasets/yl4300/Roundabout-TAU/tree/main)
- **Summariser**: [TAU-R1-Summarizer](https://huggingface.co/yl4300/TAU-R1-Summarizer)
- **Classifier**: [TAU-R1-Classifier](https://huggingface.co/yl4300/TAU-R1-Classifier)

## Training

Training follows a two-stage SFT pipeline followed by GRPO post-training, applied separately to the Classifier (Qwen3-VL-2B) and Summariser (Qwen3-VL-8B) models. Training scripts and configs are provided under `train/scripts/`. **The evaluation code will be released soon.**

## Citation
If you find this project helpful, please cite:
```bibtex
@misc{lin2026taur1visuallanguagemodel,
  title        = {TAU-R1: Visual Language Model for Traffic Anomaly Understanding},
  author       = {Yuqiang Lin and Kehua Chen and Sam Lockyer and Arjun Yadav and Mingxuan Sui and Shucheng Zhang and Yan Shi and Bingzhang Wang and Yuang Zhang and Markus Zarbock and Florain Stanek and Adrian Evans and Wenbin Li and Yinhai Wang and Nic Zhang},
  year         = {2026},
  eprint       = {2603.19098},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2603.19098}
}
```
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
