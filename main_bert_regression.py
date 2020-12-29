import argparse
from datetime import datetime
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torchtext.data import Field, TabularDataset, Iterator

from lib.utils.average_meter import AverageMeter

parser = argparse.ArgumentParser(description="Regression using deep learning")
parser.add_argument(
    "--config", help="path to configuration file", default="./config.json"
)


class BertForCoordinatesPrediction(nn.Module):
    def __init__(self, pad_index):
        super().__init__()
        self.pad_index = pad_index
        self.bert_lat = BertForSequenceClassification.from_pretrained(
            "bert-base-german-cased", num_labels=1
        )
        self.bert_long = BertForSequenceClassification.from_pretrained(
            "bert-base-german-cased", num_labels=1
        )

    def forward(self, sentences):
        masks = sentences != self.pad_index
        output_lat = self.bert_lat(input_ids=sentences, attention_mask=masks)[0]
        output_lat = output_lat.view(-1)
        output_long = self.bert_long(input_ids=sentences, attention_mask=masks)[0]
        output_long = output_long.view(-1)
        output = torch.vstack((output_lat, output_long)).T
        return output


def train(model, train_loader, criterion, optimizer, epoch, print_freq=20):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    score_lat = AverageMeter()
    score_long = AverageMeter()
    df_predictions = pd.DataFrame()

    start = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - start)

        idx, latitude, longitude, (x, _) = (
            data.id,
            data.lat,
            data.long,
            data.text,
        )
        coords = torch.vstack((latitude, longitude)).T
        predict_coords = model(x)

        loss = criterion(predict_coords, coords)
        losses.update(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for j in range(coords.size(0)):
            score_lat.update(abs(coords[j][0] - predict_coords[j][0]))
            score_long.update(abs(coords[j][1] - predict_coords[j][1]))
            df_predictions = df_predictions.append(
                {
                    "id": str(int(idx[j].item())),
                    "lat": predict_coords[j][0].item(),
                    "long": predict_coords[j][1].item(),
                },
                ignore_index=True,
            )

        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i+1}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Score {(score_lat.avg + score_long.avg)/2:.4f}"
            )
    return (score_lat.avg + score_long.avg) / 2, losses.avg, df_predictions


def validate(model, dev_loader, criterion, epoch, print_freq=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    score_lat = AverageMeter()
    score_long = AverageMeter()
    df_predictions = pd.DataFrame()

    model.eval()
    with torch.no_grad():
        start = time.time()
        for i, data in enumerate(dev_loader):
            data_time.update(time.time() - start)

            idx, latitude, longitude, (x, _) = (
                data.id,
                data.lat,
                data.long,
                data.text,
            )
            coords = torch.vstack((latitude, longitude)).T
            predict_coords = model(x)

            loss = criterion(predict_coords, coords)
            losses.update(loss.data)

            for j in range(coords.size(0)):
                score_lat.update(abs(coords[j][0] - predict_coords[j][0]))
                score_long.update(abs(coords[j][1] - predict_coords[j][1]))
                df_predictions = df_predictions.append(
                    {
                        "id": str(int(idx[j].item())),
                        "lat": predict_coords[j][0].item(),
                        "long": predict_coords[j][1].item(),
                    },
                    ignore_index=True,
                )

            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(
                    f"VALIDATION Epoch: [{epoch}][{i+1}/{len(dev_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Score {(score_lat.avg + score_long.avg)/2:.4f}"
                )
    return (score_lat.avg + score_long.avg) / 2, losses.avg, df_predictions


def test(model, test_loader, epoch, print_freq=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    df_predictions = pd.DataFrame()

    model.eval()
    with torch.no_grad():
        start = time.time()
        for i, data in enumerate(test_loader):
            data_time.update(time.time() - start)

            idx, (x, _) = data.id, data.text
            predict_coords = model(x)

            for j in range(predict_coords.size(0)):
                df_predictions = df_predictions.append(
                    {
                        "id": str(int(idx[j].item())),
                        "lat": predict_coords[j][0].item(),
                        "long": predict_coords[j][1].item(),
                    },
                    ignore_index=True,
                )

            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(
                    f"TEST Epoch: [{epoch}][{i+1}/{len(test_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                )
    return df_predictions


def main():
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.use_gpu = True
        print("Using CUDA GPU")
    else:
        args.device = torch.device("cpu")
        args.use_gpu = False
        print("Using CPU")

    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")

    MAX_LEN = 64
    PAD_INDEX = tokenizer.pad_token_id
    UNK_INDEX = tokenizer.unk_token_id
    INIT_INDEX = tokenizer.cls_token_id

    id_field = Field(sequential=False, use_vocab=False, batch_first=True)
    lat_field = Field(
        sequential=False, use_vocab=False, batch_first=True, dtype=torch.float
    )
    long_field = Field(
        sequential=False, use_vocab=False, batch_first=True, dtype=torch.float
    )
    text_field = Field(
        sequential=True,
        use_vocab=False,
        tokenize=tokenizer.encode,
        lower=False,
        include_lengths=True,
        batch_first=True,
        fix_length=MAX_LEN,
        init_token=INIT_INDEX,
        pad_token=PAD_INDEX,
        unk_token=UNK_INDEX,
    )
    fields = [
        ("id", id_field),
        ("lat", lat_field),
        ("long", long_field),
        ("text", text_field),
    ]
    fields_test = [
        ("id", id_field),
        ("text", text_field),
    ]

    train_set, dev_set = TabularDataset.splits(
        path="data/",
        train="training.txt",
        validation="validation.txt",
        format="CSV",
        fields=fields,
    )
    test_set = TabularDataset("data/test.txt", format="CSV", fields=fields_test)

    BATCH_SIZE = 16

    train_loader = Iterator(
        train_set,
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        device=args.device,
        train=True,
        sort_within_batch=True,
    )
    dev_loader = Iterator(
        dev_set,
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        device=args.device,
        train=True,
        sort_within_batch=True,
    )
    test_loader = Iterator(
        test_set,
        batch_size=BATCH_SIZE,
        device=args.device,
        train=False,
        sort=False,
    )

    model = BertForCoordinatesPrediction(pad_index=PAD_INDEX).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    results_path = f"runs/{datetime.now()}-TEST-deep-regression-BERT"
    os.mkdir(results_path)
    df_performance = pd.DataFrame()

    NUM_EPOCHS = 7
    best_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_score, train_loss, train_df_predict = train(
            model, train_loader, criterion, optimizer, epoch
        )
        print(f"END OF EPOCH {epoch} TRAIN\tscore: {train_score}\tloss: {train_loss}")
        dev_score, dev_loss, dev_df_predict = validate(
            model, dev_loader, criterion, epoch
        )
        print(f"END OF EPOCH {epoch} VALIDATION\tscore: {dev_score}\tloss: {dev_loss}")

        df_performance = df_performance.append(
            {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "train_score": train_score.item(),
                "dev_loss": dev_loss.item(),
                "dev_score": dev_score.item(),
            },
            ignore_index=True,
        )

        if best_loss > dev_loss:
            best_loss = dev_loss
            print(f"BEST EPOCH WITH LOSS {best_loss}. RUNNING ON TEST...")
            test_df_predict = test(model, test_loader, epoch)
            train_df_predict.to_csv(
                results_path + "/train_predictions.csv", index=False
            )
            dev_df_predict.to_csv(
                results_path + "/validation_predictions.csv", index=False
            )
            test_df_predict.to_csv(results_path + "/test_predictions.csv", index=False)

    df_performance.to_csv(results_path + "/performance.csv", header=True)


if __name__ == "__main__":
    main()
