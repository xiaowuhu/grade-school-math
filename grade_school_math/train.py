import torch as th
from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import datetime
import os
from torch.utils.tensorboard import SummaryWriter


def save_model(model_output_dir, model, epoch):
    path = os.path.join(model_output_dir, 'model_epoch{}'.format(epoch + 1))
    if not os.path.exists(path):
        os.makedirs(path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)

    device = th.device("cuda")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dset, batch_size=8, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 50
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    tb_writer = SummaryWriter(log_dir="tensorboard_summary/")   
    now = datetime.now()

    running_loss = 0
    overall_step = 0 # 整体的迭代计数器
    log_step = 100
    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        internal_step = 0
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs[0]
            running_loss += loss.item()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")
            if (internal_step + 1) % log_step == 0:
                running_loss = running_loss / log_step
                tb_writer.add_scalar('loss', running_loss, overall_step)
                print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                    datetime.now().hour, datetime.now().minute, 
                    internal_step + 1, epoch + 1, running_loss
                ))
                running_loss = 0
            internal_step += 1
            overall_step += 1

        save_model("c:/Gitee/model/ch7/math/", model, epoch)
    tb_writer.close()

if __name__ == "__main__":
    main()
