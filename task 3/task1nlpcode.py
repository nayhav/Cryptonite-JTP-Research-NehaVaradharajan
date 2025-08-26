import os
print("Current working directory:", os.getcwd())

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, DataCollatorForTokenClassification, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
import evaluate as eval_lib
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def seed_everything(seed=42):
 random.seed(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False

seed_everything()

model_name = "roberta-base"
batch_size = 8 if torch.cuda.is_available() else 4
max_len = 128
lr = 3e-5
epochs = 3

tokenizer = RobertaTokenizerFast.from_pretrained(model_name,add_prefix_space=True)
dataset = load_dataset("conll2003")
metric = eval_lib.load("seqeval")

tag_info = dataset["train"].features["ner_tags"].feature
tag2id = {tag: tag_info.str2int(tag) for tag in tag_info.names}
id2tag = {i: tag for tag, i in tag2id.items()}
num_labels = len(tag2id)

def encode(batch):
 encodings = tokenizer(batch["tokens"], truncation=True, padding="max_length", max_length=max_len, is_split_into_words=True, return_offsets_mapping=True)
 all_labels = []
 for i in range(len(batch["tokens"])):
  word_ids = encodings.word_ids(batch_index=i)
   word_ids = encodings.word_ids(batch_index=i)
  labels = []
  prev = None
  for idx in word_ids:
   if idx is None:
    labels.append(-100)
   elif idx != prev:
    labels.append(batch["ner_tags"][i][idx])
   else:
    labels.append(-100)
   prev = idx
  all_labels.append(labels)
   encodings["labels"] = all_labels
 return {k: encodings[k] for k in ["input_ids", "attention_mask", "labels"]}


dataset = dataset.map(encode, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

collator = DataCollatorForTokenClassification(tokenizer)
train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
val_loader = DataLoader(dataset["validation"], batch_size=batch_size, collate_fn=collator)
test_loader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=collator)

model = RobertaForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
model.config.label2id = tag2id
model.config.id2label = id2tag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=lr)
total_steps = epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

def extract_preds_labels(batch, logits):
 mask = batch["labels"] != -100
  preds = torch.argmax(logits, dim=-1)
 preds = preds[mask].cpu().numpy()
 labels = batch["labels"][mask].cpu().numpy()
 return preds, labels

def evaluate_model(loader):
 model.eval()
 all_preds, all_labels = [], []
 total = 0
 with torch.no_grad():
  for batch in loader:
   batch = {k: v.to(device) for k, v in batch.items()}
   out = model(**batch)
   total += out.loss.item()
   preds, labels = extract_preds_labels(batch, out.logits)
   grouped_preds, grouped_labels = [], []
   i = 0
   for input_ids, label_ids in zip(batch["input_ids"], batch["labels"]):
    tokens = []
    true = []
    pred = []
    for j, lid in enumerate(label_ids):
     if lid != -100:
      true.append(id2tag[lid.item()])
       pred.append(id2tag[torch.argmax(out.logits[i][j]).item()])
    grouped_preds.append(pred)
    grouped_labels.append(true)
    i += 1
   all_preds += grouped_preds
   all_labels += grouped_labels
 return total/len(loader), metric.compute(predictions=all_preds, references=all_labels)

train_loss, val_loss, val_f1 = [], [], []
best_f1 = 0

for epoch in range(epochs):
 model.train()
 total = 0
 for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
  batch = {k: v.to(device) for k, v in batch.items()}
  out = model(**batch)
  out.loss.backward()
  total += out.loss.item()
  optimizer.step()
  scheduler.step()
  optimizer.zero_grad()
  if (step+1)%100 == 0:
   print(f"Step {step+1}/{len(train_loader)} | Loss: {out.loss.item():.4f}")
 train_loss.append(total/len(train_loader))
  v_loss, result = evaluate_model(val_loader)
 val_loss.append(v_loss)
 val_f1.append(result["overall_f1"])
 print(f"Epoch {epoch+1} | Train: {train_loss[-1]:.4f} | Val: {v_loss:.4f} | F1: {val_f1[-1]:.4f}")
 if val_f1[-1] > best_f1:
  best_f1 = val_f1[-1]
  model.save_pretrained("best_model")
  tokenizer.save_pretrained("best_model")

plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

model = RobertaForTokenClassification.from_pretrained("best_model")
tokenizer = RobertaTokenizerFast.from_pretrained("best_model")
model.to(device)
_, result = evaluate_model(test_loader)

print("TEST RESULTS")
for k, v in result.items():
     if isinstance(v, dict):
        p rint(k.upper())
for kk, vv in v.items(): print(f" {kk}: {vv:.4f}")
 else:
  print(f"{k}: {v:.4f}")

def save_predictions(loader, out_path="test_predictions.conll"):
 model.eval()
 with open(out_path, "w", encoding="utf-8") as f:
  with torch.no_grad():
   for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    preds = torch.argmax(out.logits, dim=-1)
    for i in range(len(batch["labels"])):
     for j in range(len(batch["labels"][i])):
      if batch["labels"][i][j] != -100:
       tid = batch["input_ids"][i][j].item()
       tok = tokenizer.decode([tid])
       gold = id2tag[batch["labels"][i][j].item()]
       pred = id2tag[preds[i][j].item()]
       f.write(f"{tok} {gold} {pred}\n")
     f.write("\n")

save_predictions(test_loader)
if __name__ == "__main__":
     import sys
    import torch
    from transformers import pipeline
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

  
    do_inference = True 
    if do_inference:
        model_dir = "best_model" 
        tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        model = RobertaForTokenClassification.from_pretrained(model_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer,
                                aggregation_strategy="simple",
                                device=0 if torch.cuda.is_available() else -1)

        text = "Barack Obama visited New York and met Elon Musk at the Tesla HQ."

        predictions = ner_pipeline(text)
   print("\nNamed Entity Predictions:\n")
        for p in predictions:
            print(f"{p['word']} ({p['entity_group']}): score={p['score']:.4f}")

        
        def visualize_ner(text, entities):
            entity_colors = {
                "PER": "#ffcccc",  # Person
                "ORG": "#ccffcc",  # Organization
                "LOC": "#ccccff",  # Location
                "MISC": "#ffffcc"  # Misc
            }

            fig, ax = plt.subplots(figsize=(12, 2))
            ax.axis('off')
            x = 0.01
            y = 0.5
            fontsize = 12

            for i, char in enumerate(text):
                ax.text(x, y, char, fontsize=fontsize, ha='left', va='center')
                x += 0.018

            for ent in entities:
                  start, end = ent['start'], ent['end']
                label = ent['entity_group']
                color = entity_colors.get(label, "#e0e0e0")
                x_start = 0.01 + start * 0.018
                width = (end - start) * 0.018
                ax.add_patch(Rectangle((x_start, y-0.03), width, 0.06, color=color, alpha=0.4))
                ax.text(x_start, y + 0.05, label, fontsize=10, ha='left', va='bottom')

            plt.title("NER Prediction")
            plt.show()

        visualize_ner(text, predictions)
