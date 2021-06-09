class SVM_linear(LightningModule):
    def __init__(self,lr1,num_features=310,num_hidden=64,num_out=3):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor(num_features=self.hparams.num_features)
        self.model = nn.Sequential(
#             nn.Linear(self.hparams.num_features,self.hparams.num_hidden),
#             nn.ReLU(),
            nn.Linear(self.hparams.num_hidden,self.hparams.num_out)
        )
        
    def forward(self,x):
        # print("Extract features")
        # data_flat = x.view(x.size(0),-1)
        x = self.feature_extractor(x)
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        # classfication loss
        x, class_label = batch['class']
        x = x.view(x.size(0), -1)
        class_logits = self.forward(x)
        class_loss = F.cross_entropy(class_logits, class_label)
        
        #just for logging accuracy
        score, predicted = torch.max(class_logits, 1)
        acc = accuracy(predicted,class_label)
        self.log('train_acc',acc, prog_bar=True)
        self.log('train_loss', class_loss,on_epoch=True)
        
        
        return class_loss
    
    def test_step(self, batch, batch_idx):
        # validation_step defined the train loop.
        # It is independent of forward

        x, class_label = batch
        x = x.view(x.size(0), -1)
        # print(x)
        # classfication loss
        class_logits = self.forward(x)
        score, predicted = torch.max(class_logits, 1)

        class_loss = F.cross_entropy(class_logits, class_label)
        acc = accuracy(predicted,class_label)
        
        return acc
    
    def test_epoch_end(self,outputs):
        test_acc = (torch.mean(torch.stack(outputs)))
        self.test_acc=test_acc
        self.log("test_acc",test_acc)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

#         optimizer1 = torch.optim.AdamW(self.domain_discriminator.parameters(),lr=self.hparams.lr1)
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=2, gamma=0.1)
        return [optimizer], #[scheduler],scheduler1]