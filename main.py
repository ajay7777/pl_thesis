from typing import Dict
from collections import OrderedDict
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from datasets import Dataset
from transformers import AdamW
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from util import augment_train_phone_language_pairs, prep_data_pair
from transformers import TrainingArguments
import multiprocessing

MAX_LEN = 256
NUM_LABELS = 2


def get_dataloader():
    train_langs = ["en","de"]
    use_description = True

    train = pd.read_csv("datasets/pairwise_train_set_phone_medium.csv")
    train_augmented = augment_train_phone_language_pairs(train)
    test = pd.read_csv("datasets/pairwise_test_set_phone.csv")

    # Filter the train data:
    train_data = train_augmented.loc[train_augmented["lang_1"].isin(train_langs)]

    # Prepare the train and test data for the experiments
    train_data, test = prep_data_pair(train_data, test, use_description)

    train_df, val_df = train_test_split(train_data, test_size=0.2)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test)

    #train, val = train_test_split(train_data, test_size=0.3)
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case=True)

    traindataset = train_dataset.map(lambda e: tokenizer(text=e['content_1'],text_pair=e['content_2'], add_special_tokens=True, truncation=True, padding=True), batched=True)
    traindataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    testdataset = test_dataset.map(lambda e: tokenizer(text=e['content_1'],text_pair=e['content_2'], add_special_tokens=True, truncation=True, padding=True), batched=True)
    testdataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    valdataset = val_dataset.map(lambda e: tokenizer(text=e['content_1'],text_pair=e['content_2'], add_special_tokens=True, truncation=True, padding=True), batched=True)
    valdataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    # Create Trainset
    #train_set_custom = CustomDataset(train_encodings, train_data.label.tolist())
    cpu_count = multiprocessing.cpu_count()
    train_dataloader = DataLoader(
            traindataset,
            sampler=RandomSampler(val_dataset),
            batch_size=batch_size,
            num_workers=cpu_count
            )
    val_dataloader = DataLoader(
            valdataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size,
            num_workers=cpu_count
            )
    test_dataloader = DataLoader(
            testdataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=batch_size,
            num_workers=cpu_count
            )

    return train_dataloader, val_dataloader, test_dataloader


class Model(pl.LightningModule):

    def __init__(self):
        super(Model, self).__init__()
        model = 'bert-base-multilingual-uncased'
        model_config = AutoConfig.from_pretrained(model, num_labels=NUM_LABELS, return_dict = False)
        transformer_model = AutoModelForSequenceClassification.from_pretrained(model, config=model_config)

        #model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS,return_dict = False)
        self.model = transformer_model

        train_dataloader, val_dataloader, test_dataloader = get_dataloader()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=2e-5,
                )
        return optimizer

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
            })

        return output

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
            })
        return output

    def validation_end(self, outputs):
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        return result

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
            })

        return output

    def test_end(self, outputs):
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict}
        return result

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=3,
            verbose=True,
            mode="min"
            )

    trainer = pl.Trainer(
            #callbacks=[early_stop_callback],
            gpus=1,
            max_epochs =30,
            precision=16
            #auto_select_gpus = True

            # fast_dev_run = True
            # early_stop_callback=early_stop_callback,
            )
    # training_args = TrainingArguments(
    #         output_dir=f'./model',
    #         overwrite_output_dir=True,
    #         num_train_epochs=25,
    #         save_total_limit=1,
    #         per_device_train_batch_size=16,
    #         per_device_eval_batch_size=64,
    #         gradient_accumulation_steps=1,
    #         #warmup_steps=params.get('warmup_steps'),
    #         weight_decay=0.01,
    #         #evaluation_strategy="epoch",
    #         #save_strategy = params.get('evaluation_strategy'),
    #         #load_best_model_at_end=params.get('load_best_model_at_end'),
    #         metric_for_best_model="f1"
    #     )
    model = Model()

    trainer.fit(model)
    trainer.test()