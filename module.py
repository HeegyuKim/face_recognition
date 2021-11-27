
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