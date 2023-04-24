import torch
import torch.nn as nn

MLP_KERNEL_SIZE = 1
MLP_FILTERS_1 = 256
MLP_FILTERS_2 = 64
CONTRACTION_RATIO = 2
PADDING_METHOD = 'same'
UPSAMPLE_MODE = 'nearest'
N_FEATURE_EXTRACTED = 8576
N_CLASSES = 17
SEQUENCE_SIZE = 112
BATCH_SIZE = 20
BATCH_PER_EPOCH = 1000

class MLP2(nn.Module):
    def __init__(self,n_features):
        super(MLP2,self).__init__()
        self.conv1 = nn.Conv1d(n_features,MLP_FILTERS_1,MLP_KERNEL_SIZE)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(MLP_FILTERS_1,MLP_FILTERS_2,MLP_KERNEL_SIZE)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out
    
class BottleneckResBlockContract(nn.Module):
    # Takes as input a feature matrix of size:
    #  _____
    # |     |
    # |     |
    # |     |
    # |     | T
    # |     |
    # |_____|
    #    P
    # Performs a bottleneck Resnetblock:
    # BN -> ReLU -> conv(1,P) ->BN ->ReLU -> conv(3,P) -> BN -> ReLU -> conv(1,2P) -> + -> output (of size T,2P)
    # |                                                                               ^
    # L>    ->  ->  ->  ->  ->  ->  ->  -> conv(1,2P) ->  ->  ->  ->    ->  ->  ->  ->|  
    def __init__(self,in_features):
        super(BottleneckResBlockContract,self).__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_features,in_features,1)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.conv2 = nn.Conv1d(in_features,in_features,3,padding=PADDING_METHOD)
        self.bn3 = nn.BatchNorm1d(in_features)
        self.conv3 = nn.Conv1d(in_features,in_features*CONTRACTION_RATIO,1)
        self.conv_skip = nn.Conv1d(in_features,in_features*CONTRACTION_RATIO,1)

    def forward(self,x):
        y = self.bn1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.conv3(y)
        z = self.conv_skip(x)
        return y+z
    
class BottleneckResBlockExpand(nn.Module):
    # Takes as input a feature matrix of size:
    #  _____
    # |     |
    # |     |
    # |     |
    # |     | T
    # |     |
    # |_____|
    #    P
    # Performs a bottleneck Resnetblock:
    # BN -> ReLU -> conv(1,P/2) ->BN ->ReLU -> conv(3,P/2) -> BN -> ReLU -> conv(1,P/2) -> + -> output (of size T,P/2)
    # |                                                                               ^
    # L>    ->  ->  ->  ->  ->  ->  ->  -> conv(1,P/2) ->  ->  ->  ->    ->  ->  ->  ->|  
    def __init__(self,in_features):
        in_features = int(in_features)
        super(BottleneckResBlockExpand,self).__init__()
        self.bn1 = nn.BatchNorm1d(int(in_features))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_features,int(in_features/CONTRACTION_RATIO),1)
        self.bn2 = nn.BatchNorm1d(int(in_features/CONTRACTION_RATIO))
        self.conv2 = nn.Conv1d(int(in_features/CONTRACTION_RATIO),int(in_features/CONTRACTION_RATIO),3,padding=PADDING_METHOD)
        self.bn3 = nn.BatchNorm1d(int(in_features/CONTRACTION_RATIO))
        self.conv3 = nn.Conv1d(int(in_features/CONTRACTION_RATIO),int(in_features/CONTRACTION_RATIO),1)
        self.conv_skip = nn.Conv1d(in_features,int(in_features/CONTRACTION_RATIO),1)

    def forward(self,x):
        y = self.bn1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.conv3(y)
        z = self.conv_skip(x)
        return y+z

class UpsamplingBlock(nn.Module):
# Takes as input a feature matrix of size (with ncol: nFeatures and nrow: nTimePoints) :
    #  _____
    # |     |
    # |     |
    # |     |
    # |     | T
    # |     |
    # |_____|
    #    P
# Upsample uses nearest neighbor to output a 2TxP matrix
# a Conv(2,P/2) is then applied with padding = same
    
    def __init__(self,in_feature):
        super(UpsamplingBlock,self).__init__()
        in_feature = int(in_feature)
        self.upsample = nn.Upsample(scale_factor=2,mode=UPSAMPLE_MODE)
        self.conv = nn.Conv1d(in_feature, int(in_feature/2),kernel_size=2, padding=PADDING_METHOD)
    
    def forward(self,x):
        out = self.upsample(x)
        out = self.conv(out)
        return out
    

class Unet1D(nn.Module):
    def __init__(self,in_feature):
        super(Unet1D,self).__init__()

        #LEFT PART
        self.maxpool = nn.MaxPool1d(2)

        self.contr1 = BottleneckResBlockContract(in_feature)
        #self.maxpool
        nb_f = in_feature*2
        self.contr2 = BottleneckResBlockContract(nb_f)
        #self.maxpool
        nb_f = nb_f*2
        self.contr3 = BottleneckResBlockContract(nb_f)
        #self.maxpool
        nb_f = nb_f*2
        #self.contr4 = BottleneckResBlockContract(nb_f)
        #self.maxpool
        #nb_f = nb_f*2
        ##############################################

        #DOWN PART
        
        self.contr5 = BottleneckResBlockContract(nb_f)
        nb_f = nb_f*2
        ##############################################

        #RIGHT PART

        #self.upsample4 = UpsamplingBlock(nb_f)
        #nb_f = nb_f/2
        #concat
        #nb_f = nb_f*2
        #self.expand4 = BottleneckResBlockExpand(nb_f)
        #nb_f = nb_f/2

        self.upsample3 = UpsamplingBlock(nb_f)
        nb_f = nb_f/2
        #concat
        nb_f = nb_f*2
        self.expand3 = BottleneckResBlockExpand(nb_f)
        nb_f = nb_f/2

        self.upsample2 = UpsamplingBlock(nb_f)
        nb_f = nb_f/2
        #concat
        nb_f = nb_f*2
        self.expand2 = BottleneckResBlockExpand(nb_f)
        nb_f = nb_f/2

        self.upsample1 = UpsamplingBlock(nb_f)
        nb_f = nb_f/2
        #concat
        nb_f = nb_f*2
        self.expand1 = BottleneckResBlockExpand(nb_f)
        nb_f = nb_f/2

        self.upsample0 = UpsamplingBlock(nb_f)
        nb_f = nb_f/2
        #concat
        nb_f = nb_f*2
        self.expand0 = BottleneckResBlockExpand(nb_f)
        nb_f = nb_f/2

    def forward(self,x):
        # X is of shape [nbatch,64,224]

        # LEFT PART
        l0 = x #[P,T]           #Output shape
        a = self.maxpool(l0)    #[P,T/2]
        l1 = self.contr1(a)     #[P*2,T/2]
        b = self.maxpool(l1)    #[P*2,T/4]
        l2 = self.contr2(b)     #[P*4,T/4]
        c = self.maxpool(l2)    #[P*4,T/8]
        l3 = self.contr3(c)     #[P*8,T/8]
        d = self.maxpool(l3)    #[P*8,T/16]
        #l4 = self.contr4(d)     #[P*16,T/16]
        #e = self.maxpool(l4)    #[P*16, T/32]

        # DOWN PART
        l5 = self.contr5(d)     #[P*32,T/32]

        # RIGHT PART
        #r4 = self.upsample4(l5)     #[P*16,T/16]
        #t4 = torch.cat((l4,r4),1)   #[P*32,T/16]
        #t4 = self.expand4(t4)       #[P*16,T/16]
        r3 = self.upsample3(l5)     #[P*8,T/8]
        t3 = torch.cat((l3,r3),1)   #[P*16,T/8]
        t3 = self.expand3(t3)       #[P*8,T/8]
        r2 = self.upsample2(t3)     #[P*4,T/4]
        t2 = torch.cat((l2,r2),1)   #[P*8,T/4]
        t2 = self.expand2(t2)       #[P*4,T/4]
        r1 = self.upsample1(t2)     #[P*2,T/2]
        t1 = torch.cat((l1,r1),1)   #[P*4,T/2]
        t1 = self.expand1(t1)       #[P*2,T/2]
        r0 = self.upsample0(t1)     #[P,T]
        t0 = torch.concat((l0,r0),1)#[P*2,T]
        t0 = self.expand0(t0)       #[P,T]

        return t0 # P features, T points


class PredictionHead(nn.Sequential):
    def __init__(self,n_features):
        super(PredictionHead,self).__init__()
        self.mlp = MLP2(n_features)
        self.unet = Unet1D(MLP_FILTERS_2)
        self.convHead = nn.Conv1d(MLP_FILTERS_2,N_CLASSES,kernel_size=3,padding=PADDING_METHOD)

class TemporalHead(nn.Sequential):
    def __init__(self,n_features):
        super(TemporalHead,self).__init__()   
        self.mlp = MLP2(n_features)
        self.unet = Unet1D(MLP_FILTERS_2)
        self.convHead = nn.Conv1d(MLP_FILTERS_2,N_CLASSES,kernel_size=3,padding=PADDING_METHOD)

class Network():
    def __init__(self):
        self.displacements = TemporalHead(N_FEATURE_EXTRACTED)
        self.confidences = PredictionHead(N_FEATURE_EXTRACTED)

    def __init__(self, path_confidence, path_displacement) -> None:
        self.displacements = TemporalHead(N_FEATURE_EXTRACTED)
        self.confidences = PredictionHead(N_FEATURE_EXTRACTED)
        self.displacements.load_state_dict(torch.load(path_displacement, map_location=torch.device('cpu')))
        self.confidences.load_state_dict(torch.load(path_confidence,map_location=torch.device('cpu')))
    
    def get_confidence_model(self):
        return self.confidences
    
    def get_displacement_model(self):
        return self.displacements
    
    def get_models(self):
        return self.confidences, self.displacements
