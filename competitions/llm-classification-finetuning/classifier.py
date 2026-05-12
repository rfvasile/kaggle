import argparse
import os
import random
from argparse import Namespace
from datetime import datetime
from typing import Any, override

import kagglehub
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor, optim, tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorWithPadding,
                          DebertaForSequenceClassification, DebertaV2Tokenizer)


def get_reproducible_dataloader_kwargs(seed: int = 0) -> dict[str, Any]:
    """
    Returns a dictionary of arguments to pass to a DataLoader
    to ensure worker-level reproducibility.
    """
    def seed_worker(_: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return {
        "worker_init_fn": seed_worker,
        "generator": g,
    }



def make_reproducible(seed: int = 0) -> None:

    """
    Sets all seeds and configuration flags for reproducibility.
    See: https://docs.pytorch.org/docs/2.11/notes/randomness.html
    """
    # 1. Basic Python and NumPy seeding
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. PyTorch seeding (CPU and all GPUs)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 3. Algorithm Determinism
    # This will throw an error if an operation doesn't have a deterministic implementation
    torch.use_deterministic_algorithms(True)

    # Required for certain CUDA operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 4. CuDNN Backend settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Reproducibility set with seed: {seed}")


class LLMClassificationDataset(Dataset):
    def __init__(self, dataset: DataFrame, tokenizer: DebertaV2Tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.texts = (
            dataset["prompt"].astype("str")
            + " [SEP] "
            + dataset["response_a"].astype("str")
            + " [SEP] "
            + dataset["response_b"].astype("str")
        ).to_list()
        self.labels = (
            dataset[["winner_model_a", "winner_model_b", "winner_tie"]]
            .values.argmax(axis=1)
            .tolist()
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:

        encoding: dict[str, Any] = self.tokenizer(
            self.texts[index],
            truncation=False,
            padding=False,  # no padding here, collator handles it
            # collator also handles tensor transform
            # return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": tensor(self.labels[index], dtype=torch.long)
        }



class DebertaClassifier(Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
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


def get_train_dataset(
    dataset_id: str,
    model_id: str,
) -> tuple[LLMClassificationDataset, DebertaV2Tokenizer]:
    """Read the competition dataset."""
    # Read pandas dataframe
    path = kagglehub.competition_download(dataset_id)
    print(path)
    kaggle_dataset = pd.read_csv(f"{path}/train.csv")

    # Get pytorch standard format
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert isinstance(tokenizer, DebertaV2Tokenizer), "Unexpected type"

    dataset = LLMClassificationDataset(kaggle_dataset, tokenizer)
    return dataset, tokenizer


def get_loader(
        dataset: Dataset, tokenizer: DebertaV2Tokenizer, shuffle: bool, batch_size: int = 2
) -> DataLoader:
    """Dataset loader for train/validation splits."""
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt", padding=True),
        shuffle=shuffle,
        num_workers=4,
        **get_reproducible_dataloader_kwargs(42)
    )
    return data_loader


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
        criterion: Module,
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def train_one_epoch(
        self,
        epoch_index: int,
        tb_writer: SummaryWriter
    ) -> None:
        accumulated_loss = 0.0
        print_freq = 2

        for i, data in enumerate(self.training_loader):
            labels = data.pop("labels").to(self.device)
            inputs = {k: v.to(self.device) for k, v in data.items()}

            self.optimizer.zero_grad()

            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
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
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)

                accumulated_loss = 0

        return last_loss

    def train(self, num_epochs: int) -> None:
        """Training loop"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
                    vloss = self.criterion(vlogits, vlabels)
                    running_vloss += vloss.item()


            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train {avg_loss} valid {avg_vloss}")

            writer.add_scalars("Training vs Validation Loss",
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               current_epoch + 1)

            writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                path = f"model_{timestamp}_{current_epoch}"
                torch.save(self.model.state_dict(), path)





def main(args: Namespace, seed: int = 42) -> None:
    """Main training function."""
    make_reproducible(seed)

    # Dataset components
    full_dataset, tokenizer = get_train_dataset(args.dataset_id, args.model_id)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    training_loader = get_loader(train_dataset, tokenizer, shuffle=True)
    val_loader = get_loader(val_dataset, tokenizer, shuffle=False)


    # Network components
    model = DebertaClassifier(model_name=args.model_id, num_labels=args.num_labels)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    trainer = Trainer(training_loader, val_loader, model, optimizer, criterion)
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="llm-classification-finetuning")
    parser.add_argument("--model-id", default="microsoft/deberta-v3-small")
    parser.add_argument("--num-labels", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=10)

    args = parser.parse_args()

    main(args)
