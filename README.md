# IMDb Comment Classifier 🎬🧠

This project is a sentiment classification model built using the Hugging Face Transformers library. It leverages a pre-trained BERT model (`bert-base-uncased`) to classify IMDb movie reviews as **positive** or **negative**.

## 🗂️ Dataset

- **Source**: [IMDb Movie Reviews](https://huggingface.co/datasets/imdb)
- **Size**: 50,000 reviews (25k training, 25k test)
- **Labels**: Binary sentiment (`positive` or `negative`)

## 🚀 Model

- **Base Model**: `bert-base-uncased` from Hugging Face
- **Fine-tuned for**: Binary sequence classification
