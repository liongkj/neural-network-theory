# Here are some suggestions of parameter settings. The feature extractor has 2 layers, # both with node number of 128.

class SourceFeatureExtractor(LightningModule):
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

class TargetFeatureExtractor(LightningModule):
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
    
# class ADDA_step2(LightningModule):
#     def __init__(self,lr,num_features=310 ,num_class=3)
#         super().__init__()

#         self.save_hyperparameters()
#         self.domain_discriminator = DomainDiscriminator()
        
#     def forward(self, x):
#         # in lightning, forward defines the prediction/inference actions
        
#         target_features = self.target_feature_extractor(x)
#         # score, predicted = torch.max(pred, 1)
#         return target_features
    
#     def training_step(self, batch, batch_idx):

class ADDA_step1(LightningModule):
    def __init__(self,lr,num_features=310 ,num_class=3):
        super().__init__()

        self.save_hyperparameters()
        self.source_feature_extractor = SourceFeatureExtractor(num_features=self.hparams.num_features)
        self.target_feature_extractor = TargetFeatureExtractor(num_features=self.hparams.num_features)
        self.label_predictor = LabelPredictor(num_class=self.hparams.num_class)
        self.domain_discriminator = DomainDiscriminator()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        source_features = self.source_feature_extractor(x)
        target_features = self.target_feature_extractor(x)
        return source_features,target_features

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop.
        # It is independent of forward
        loss = 0
        if (self.current_epoch < 5):
            if optimizer_idx==0:
            # classfication loss
                self.target_feature_extractor.freeze()
                x, class_label = batch['class']
                x = x.view(x.size(0), -1)
                source_features,target_features = self.forward(x)
                class_logits = self.label_predictor(source_features)
                class_loss = F.cross_entropy(class_logits, class_label)
                loss+=class_loss
                #just for logging accuracy
                score, predicted = torch.max(class_logits, 1)
                acc = accuracy(predicted,class_label)
                # self.log('train_acc',acc, prog_bar=True)
            #domain loss              
            
        # Logging to TensorBoard by default
        self.log('train_loss', loss,on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.source_feature_extractor.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.target_feature_extractor.parameters(), lr=1e-5)
        return [g_opt, d_opt]
    
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

class ADDA_step2(LightningModule):
    def __init__(self,lr,num_features=310 ,num_class=3):
        super().__init__()

        self.save_hyperparameters()
        self.source_feature_extractor = SourceFeatureExtractor(num_features=self.hparams.num_features)
        self.target_feature_extractor = TargetFeatureExtractor(num_features=self.hparams.num_features)
        self.label_predictor = LabelPredictor(num_class=self.hparams.num_class)
        self.domain_discriminator = DomainDiscriminator()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        source_features = self.source_feature_extractor(x)
        target_features = self.target_feature_extractor(x)
        return source_features,target_features

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop.
        # It is independent of forward
        loss = 0
        if (self.current_epoch < 5):
            if optimizer_idx==0:
            # classfication loss
                self.target_feature_extractor.freeze()
                x, class_label = batch['class']
                x = x.view(x.size(0), -1)
                source_features,target_features = self.forward(x)
                class_logits = self.label_predictor(source_features)
                class_loss = F.cross_entropy(class_logits, class_label)
                loss+=class_loss
                #just for logging accuracy
                score, predicted = torch.max(class_logits, 1)
                acc = accuracy(predicted,class_label)
                # self.log('train_acc',acc, prog_bar=True)
            #domain loss
        # else: #training discriminator after n epoch
        #     if(optimizer_idx==1):
        #     # do training_step with decoder
        #         self.source_feature_extractor.freeze()
        #         domain_data, domain_label = batch['domain']
        #         domain_data = domain_data.view(domain_data.size(0), -1)

        #         domain_features = self.forward(domain_data)
        #         domain_logits = self.domain_discriminator(domain_features)
        #         # print(domain_logits)
        #         # input()
        #         domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_label)
        #         loss-=domain_loss
        #         predicted_domain = torch.round(torch.sigmoid(domain_logits)).int()
        #         # print(predicted_domain)
        #         # print(f'domain_label: {domain_label}')
        #         # input()
        #         domain_acc = accuracy(predicted_domain,domain_label.int())
        #         self.log('domain_acc',domain_acc,prog_bar=True)
        #         self.log('domain_loss',domain_loss,prog_bar=True)
                
            
        # Logging to TensorBoard by default
        self.log('train_loss', loss,)
        return loss
    
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.source_feature_extractor.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.target_feature_extractor.parameters(), lr=1e-5)
        return [g_opt, d_opt]
    
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