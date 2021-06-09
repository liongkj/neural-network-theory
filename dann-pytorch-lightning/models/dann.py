"""# Dann model"""

# Here are some suggestions of parameter settings. The feature extractor has 2 layers, # both with node number of 128.

class FeatureExtractor(LightningModule):
    def __init__(self,num_features=310,num_hidden=128,num_out=64):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(self.hparams.num_features,self.hparams.num_hidden),
            nn.ReLU(),
            nn.Linear(self.hparams.num_hidden,self.hparams.num_out)
        )
        
    def forward(self,x):
        # print("Extract features")
        # data_flat = x.view(x.size(0),-1)
        out = self.model(x)
        return out


# The label predictor and domain discriminator have 3
# layers with node numbers of 64, 64, and C, respectively. C indicates the number of
# emotion classes to be classified.

class LabelPredictor(LightningModule):
    def __init__(self,num_hidden=64,num_class=3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden),
            nn.ReLU(),
            nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden),
            nn.ReLU(),
            nn.Linear(self.hparams.num_hidden, self.hparams.num_class)
        )

    def forward(self,x):
        # print("Predict label")
        out = self.model(x)
        return out


class DomainDiscriminator(LightningModule):
    def __init__(self,num_hidden=64,num_class=1):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden),
            nn.ReLU(),
            nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(self.hparams.num_hidden, self.hparams.num_class)
        )
    def forward(self,x):
        # print("Predict domain label")
        out = self.model(x)
        # print(out)
        # print(f'out shape:{out.shape}')
        return out

class DANN(LightningModule):
    def __init__(self,lr1,lr2,num_features=310 ,num_class=3):
        super().__init__()

        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor(num_features=self.hparams.num_features)
        self.label_predictor = LabelPredictor(num_class=self.hparams.num_class)
        # self.loss_func = nn.CrossEntropyLoss()
        self.domain_discriminator = DomainDiscriminator()
        self.test_acc = 0
        self.test_epoch = 0

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        features = self.feature_extractor(x)
        # score, predicted = torch.max(pred, 1)
        return features

    def training_step(self, batch, batch_idx,optimizer_idx):
        # training_step defined the train loop.
        # It is independent of forward

        loss = 0
        # classfication loss
        if optimizer_idx == 0:
            
            x, class_label = batch['class']
            x = x.view(x.size(0), -1)
            features = self.forward(x)
            class_logits = self.label_predictor(features)
            class_loss = F.cross_entropy(class_logits, class_label)
            loss+=class_loss
            #just for logging accuracy
            score, predicted = torch.max(class_logits, 1)
            acc = accuracy(predicted,class_label)
            self.log('train_acc',acc, prog_bar=True)
        #domain loss
        
        if optimizer_idx == 1:
        # do training_step with decoder
            domain_data, domain_label = batch['domain']
            domain_data = domain_data.view(domain_data.size(0), -1)

            domain_features = self.forward(domain_data)
            domain_logits = self.domain_discriminator(domain_features)
            # print(domain_logits)
            # input()
            domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_label)
            loss-=domain_loss
            predicted_domain = torch.round(torch.sigmoid(domain_logits)).int()
            # print(predicted_domain)
            # print(f'domain_label: {domain_label}')
            # input()
            domain_acc = accuracy(predicted_domain,domain_label.int())
            self.log('domain_acc',domain_acc,prog_bar=True)
            self.log('domain_loss',domain_loss,prog_bar=True)

        

        # Logging to TensorBoard by default
        self.log('train_loss', loss,on_epoch=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # validation_step defined the train loop.
        # It is independent of forward

        x, class_label = batch
        x = x.view(x.size(0), -1)
        # print(x)
        # classfication loss
        features = self.forward(x)
        class_logits = self.label_predictor(features)
        score, predicted = torch.max(class_logits, 1)

        class_loss = F.cross_entropy(class_logits, class_label)
        acc = accuracy(predicted,class_label)
        
        return acc

    def test_epoch_end(self,outputs):
        test_acc = (torch.mean(torch.stack(outputs)))
        self.test_acc=test_acc
        self.log("test_acc",test_acc)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

        optimizer1 = torch.optim.AdamW(self.domain_discriminator.parameters(),lr=self.hparams.lr1)
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=2, gamma=0.1)
        return [optimizer,optimizer1], [scheduler]#,scheduler1]