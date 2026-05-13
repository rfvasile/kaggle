import argparse
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, override

import kagglehub
import pandas as pd
import torch
from competitions.utils import (get_reproducible_dataloader_kwargs,
                                make_reproducible)
from pandas import DataFrame
from torch import Tensor, optim, stack, tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorWithPadding,
                          DebertaForSequenceClassification, DebertaV2Tokenizer)


@dataclass
class DataCollatorForSeparateClassification:
    def __init__(self, collator: Callable[[list[Any]], Any], indices: list[str]) -> None:
        self.default_collator = collator
        self.nested_keys = indices

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {}
        for key in self.nested_keys:
            data = [s[key] for s in features]
            batch[key] = self.default_collator(data)

        non_nested_keys = [k for k in features[0] if k not in self.nested_keys]
        for key in non_nested_keys:
            data = [s[key] for s in features]
            batch[key] = stack(data)

        return batch


class LLMClassificationDataset(Dataset):
    def __init__(self, dataset: DataFrame, tokenizer: DebertaV2Tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["prompt"].astype("str")
        self.labels = (
            dataset[["winner_model_a", "winner_model_b", "winner_tie"]]
            .values.argmax(axis=1)
            .tolist()
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def build_text(prompt: str, response: str) -> str:
        return f"{prompt} [SEP] {response}"

    def __getitem__(self, index: int) -> dict[str, Any]:

        winner_label = self.labels[index]
        sample = self.dataset.loc[index]

        prompt = str(sample["prompt"])
        response_a = str(sample["response_a"])
        response_b = str(sample["response_b"])

        if winner_label == 0:
            chosen_sample = self.build_text(prompt, response_a)
            rejected_sample = self.build_text(prompt, response_b)
        else:
            chosen_sample = self.build_text(prompt, response_b)
            rejected_sample = self.build_text(prompt, response_a)

        tokenizer_kwargs = {"truncation": False, "padding": False}
        chosen_encoding: dict[str, Any] = self.tokenizer(
            chosen_sample,
            **tokenizer_kwargs
        )

        rejected_encoding: dict[str, Any] = self.tokenizer(
            rejected_sample,
            **tokenizer_kwargs
        )
        return {
            "chosen_batch": chosen_encoding,
            "rejected_batch": rejected_encoding,
            "labels": tensor(winner_label, dtype=torch.long),
        }


class DebertaClassifier(Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 1,
        problem_type: str = "single_label_classification",
    ) -> None:
        super(DebertaClassifier, self).__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type=problem_type
        )

    @override
    def forward(self, x: dict[str, Any]) -> Tensor:
        output = self.classifier(
            input_ids=x["input_ids"], attention_mask=x["attention_mask"]
        )

        return output.logits



class Trainer:
    """
    Base Trainer.
    See: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
    """

    def __init__(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        model: Module,
        optimizer: optim.Optimizer,
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def train_one_epoch(self, epoch_index: int, tb_writer: SummaryWriter) -> float:
        accumulated_loss = 0.0
        print_freq = 2

        for i, data in enumerate(self.training_loader):
            labels = data.pop("labels").to(self.device)
            chosen_batch = data.pop("chosen_batch")
            rejected_batch = data.pop("rejected_batch")
            chosen_batch = {k: v.to(self.device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(self.device) for k, v in rejected_batch.items()}

            self.optimizer.zero_grad()

            chosen_scores = self.model(chosen_batch)
            rejected_scores = self.model(rejected_batch)

            loss = self.loss_fn(chosen_scores, rejected_scores, labels)
            loss.backward()

            self.optimizer.step()
            accumulated_loss += loss.item()

            # We print data every 1000 steps
            if i % print_freq == print_freq - 1:
                # Calculate the average loss
                last_loss = accumulated_loss / print_freq
                print(f"    batch {i + 1} loss: {last_loss}")

                # Add data to tensorboard
                tb_x = epoch_index * len(self.training_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)

                accumulated_loss = 0

        return last_loss

    def train(self, num_epochs: int) -> None:
        """Training loop"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f"runs/llm_classification_finetuning_{timestamp}")

        best_vloss = 1_000_000

        for current_epoch in range(num_epochs):
            print(f"Epoch {current_epoch + 1}:")

            self.model.train(True)
            avg_loss = self.train_one_epoch(current_epoch, writer)

            running_vloss = 0.0
            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    print(f"Validation batch {i}")
                    vlabels = vdata.pop("labels").to(self.device)
                    vinputs = {k: v.to(self.device) for k, v in vdata.items()}
                    vlogits = self.model(vinputs)
                    vloss = self.loss_fn(vlogits, vlabels, vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train {avg_loss} valid {avg_vloss}")

            writer.add_scalars(
                "Training vs Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                current_epoch + 1,
            )

            writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                path = f"model_{timestamp}_{current_epoch}"
                torch.save(self.model.state_dict(), path)

    def loss_fn(self, chosen_scores: Tensor, rejected_scores: Tensor, labels: Tensor) -> Tensor:
        loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()
        return loss


def get_train_dataset(
    dataset_id: str = "llm-classification-finetuning",
    model_id: str = "microsoft/deberta-v3-small",
) -> tuple[LLMClassificationDataset, DebertaV2Tokenizer, DataFrame]:
    """Read the competition dataset."""
    # Read pandas dataframe
    path = kagglehub.competition_download(dataset_id)
    kaggle_dataset = pd.read_csv(f"{path}/train.csv")

    # Get pytorch standard format
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert isinstance(tokenizer, DebertaV2Tokenizer), "Unexpected type"

    dataset = LLMClassificationDataset(kaggle_dataset, tokenizer)
    return dataset, tokenizer, kaggle_dataset


def get_loader(
    dataset: Dataset, tokenizer: DebertaV2Tokenizer, shuffle: bool, batch_size: int = 2
) -> DataLoader:
    """Dataset loader for train/validation splits."""
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt", padding=True)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForSeparateClassification(
            collator, ["chosen_batch", "rejected_batch"]
        ),
        shuffle=shuffle,
        num_workers=4,
        **get_reproducible_dataloader_kwargs()
    )
    return data_loader


def main(args: Namespace, seed: int = 42) -> None:
    """Main training function."""
    make_reproducible(seed)

    # Dataset components
    full_dataset, tokenizer, _ = get_train_dataset(args.dataset_id, args.model_id)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    training_loader = get_loader(train_dataset, tokenizer, shuffle=True)
    val_loader = get_loader(val_dataset, tokenizer, shuffle=False)

    # Network components
    model = DebertaClassifier(model_name=args.model_id, num_labels=args.num_labels)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)

    trainer = Trainer(training_loader, val_loader, model, optimizer)
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="llm-classification-finetuning")
    parser.add_argument("--model-id", default="microsoft/deberta-v3-small")
    parser.add_argument("--num-labels", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=10)

    args = parser.parse_args()

    main(args)
