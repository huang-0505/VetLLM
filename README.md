# 🐾 VetLLM

**VetLLM** is a domain-specific large language model (LLM) fine-tuned on publicly available veterinary medical literature, including peer-reviewed journal articles and case studies. It is designed to answer questions related to veterinary science, diagnostics, treatments, and pet health.

### 🧠 Model Highlights
- **Base model**: [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- **Fine-tuning**: Performed using [LoRA](https://arxiv.org/abs/2106.09685) adapters for efficiency
- **Training dataset**: 146M tokens of veterinary-focused text, curated from public scientific sources
- **Training infrastructure**: 8× NVIDIA H100 GPUs on Google Cloud Platform (GCP)
- **Total training time**: 70 hours

---

🛠️ Setup
Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```
📁 Project Structure

```base
VetLLM/
├── vetllm.py                        # Training and inference script
├── combined.txt                    # Raw training corpus
├── results/                        # Checkpoints and logs
├── finetuned-mistral7b-vet-lora/   # LoRA adapter output
├── finetuned-mistral7b-vet-merged/ # Final merged model
├── run.sh                          # Shell script for GCP automation
├── requirements.txt
└── README.md
```


📜 License
This project uses publicly accessible veterinary articles. Verify licensing before commercial use or redistribution.
