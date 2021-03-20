from transformers import BertConfig, BertModel, BertTokenizer, BertForMaskedLM
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from tokenizers import Tokenizer
# from Dataset import load_dataset
from torch.utils.data import DataLoader
import build_dataset


device = "cuda"
filepath = '/home/ubuntu/BERT-GAN/BERT_GAN/master/master_train.txt'
vocab_size = 249
batch_size = 128


dataset = build_dataset.TrainDataset(filepath)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


config = BertConfig(
    vocab_size=249,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = Tokenizer.from_file("/home/ubuntu/BERT-GAN/BERT_GAN/bAbI_tokenizer.json")

model = BertForMaskedLM(config=config)

training_args = TrainingArguments(
        output_dir="/home/ubuntu/BERT-GAN/BERT_GAN/bAbibert_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
)

# %%time
trainer.train()

