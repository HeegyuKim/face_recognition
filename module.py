
def output_keys(outputs, key):
  return torch.stack([x[key] for x in outputs])

class ArcFaceNetModule(pl.LightningModule):
  def __init__(self, num_classes):
    super().__init__()
    self.num_classes = num_classes
    self.model = ArcFaceNet(num_classes)
    self.criterion = ArcFaceLoss(crit="focal")

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
    return optimizer

  def training_step(self, train_batch, batch_idx):
    y_p, loss, acc = self.step(train_batch)
    # self.log('train_loss', loss)
    return { "loss": loss, "acc": acc}

  def training_epoch_end(self, outputs):
    self.log_dict({
        "avg_loss": output_keys(outputs, "loss").mean(),
        "train_acc": output_keys(outputs, "acc").mean()
    })

  # def validation_step(self, valid_batch, batch_idx):
  #   batch['embedding'] = self(batch['image'])['embeddings']
  #   return {}
  
  # def validation_epoch_end(self, outputs):
    

  def step(self, batch):
    logits = self(batch['image'])['logits']
    loss = arcface_loss_fn(self.criterion, logits, batch['label'], self.num_classes)
    acc = (torch.argmax(logits, dim=1) == batch['label']).double().mean()

    return logits, loss, acc




class FashionMnistModule(pl.LightningModule):
  def __init__(self, model, criterion, config, logger=None):
    super().__init__()
    self.model = model
    self.criterion = criterion
    self.config = config

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    # https://sanghyu.tistory.com/113
    optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
    return [optimizer], [lr_scheduler]

  def train_dataloader(self):
    train_data = dset.FashionMNIST(root=root, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_data, 
                             batch_size=self.config['batch_size'], 
                              shuffle=True,
                              drop_last=True)
    return train_loader

  def val_dataloader(self):
    test_data = dset.FashionMNIST(root=root, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, 
                             batch_size=self.config['batch_size'], 
                             shuffle=False,
                             drop_last=True)
    return test_loader

  def step(self, batch, idx, type):
    output = self(batch[0])
    embs = output['embeddings']
    logits = output['logits']
    
    labels = batch[1]

    loss = self.criterion(logits, labels)
    acc = torchmetrics.functional.accuracy(logits, labels)

    self.log_dict({
        f"{type}_loss": loss,
        f"{type}_accuracy": acc
    })

    return { "loss": loss, "accuracy": acc, 
            "embeddings": embs.detach(),
            "labels": labels}

  def epoch_end(self, outputs, type):
    mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
    mean_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
    embeddings = torch.concat([x['embeddings'] for x in outputs])
    labels = torch.concat([x['labels'] for x in outputs])

    self.log_dict({
        f"{type}_mean_loss": mean_loss,
        f"{type}_mean_accuracy": mean_acc
    })

    save_umap_plot(embeddings.cpu(), labels.cpu())
    plt.savefig("tmp.png")
    plt.cla()
    self.logger.experiment.log_image("tmp.png", name=f"PCA-{type}")

  def training_step(self, batch, idx):
    return self.step(batch, idx, "train")

  def training_epoch_end(self, outputs):
    self.epoch_end(outputs, "train")

  def validation_step(self, batch, idx):
    return self.step(batch, idx, "val")

  def validation_epoch_end(self, outputs):
    return self.epoch_end(outputs, "val")
