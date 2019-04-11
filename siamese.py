import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import img_to_vec
# print("correct file")
# torch.cuda.set_device(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
model = models.resnet18(pretrained=True)
model.to(device)
model.eval()
for params in model.parameters():
    params.requires_grad = False
layer = model._modules.get('layer2')



def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = "data/train_2/"
    testing_dir = "data/test/"
    custom_test_dir = "data/custom_test/"
    train_batch_size = 64
    train_number_epochs = 100

def get_vector(image_name, is_path=True):
    
    if is_path:
        img = Image.open(image_name)
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    else:
        img = image_name
        t_img = Variable(normalize(img).unsqueeze(0))
    # t_img.to(device)
    my_embedding = []
    
    def copy_data(m, i, o):
        my_embedding.append(o.data)
    h = layer.register_forward_hook(copy_data)
    t_img = t_img.to(device)
    model(t_img)
    h.remove()
    return my_embedding

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True,
                 is_test=False,pick_similar_samples=True,
                is_custom_test=False):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.is_test = is_test
        self.pick_similar_samples = pick_similar_samples
        self.is_custom_test = is_custom_test
        
    def __getitem__(self,index):
        if not self.is_custom_test:
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            #we need to make sure approx 50% of images are in the same class
            if not self.is_test:
                should_get_same_class = random.randint(0,1)
            else:
                if self.pick_similar_samples : should_get_same_class = 1
                else: should_get_same_class = 0

            if should_get_same_class:
                while True:
                    #keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                    if img0_tuple[1]==img1_tuple[1]:
                        break
            else:
                while True:
                    #keep looping till a different class image is found

                    img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                    if img0_tuple[1] !=img1_tuple[1]:
                        break
        else:
            img0_tuple = (self.imageFolderDataset.imgs[0])
            img1_tuple = (self.imageFolderDataset.imgs[1])
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
#         print("ïmg 0 shape: {0}".format(img0.shape))
        img0_vec = get_vector(img0_tuple[0], is_path=True)[0]
        img1_vec = get_vector(img1_tuple[0], is_path=True)[0]
        img0_vec.squeeze_()
        img1_vec.squeeze_()
#         print("ïmg 0 shape: {0}".format(img0_vec.shape))
        return img0, img1, img0_vec, img1_vec , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),


            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(16*28*28, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
#         print("1: {0}".format(output.shape))
        output = output.view(output.size()[0], -1)
#         print("2: {0}".format(output.shape))
        output = self.fc1(output)
#         print("3: {0}".format(output.shape))
        return output

    def forward(self, input1, input2):
        #print(input1)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def main():
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((224,224)),
                                                                        transforms.ColorJitter(brightness=0.1,contrast=0.1),
                                                                        transforms.Grayscale(num_output_channels=1),
                                                                        transforms.RandomRotation([0,75]),
                                                                        transforms.RandomAffine([0,20], translate=(0.1, 0.95), scale=(0.5,2), shear=5),
                                                                        transforms.ToTensor()
                                                                        ])
                                        ,should_invert=False)

    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=Config.train_batch_size)
    print(len(train_dataloader))
    net = SiameseNetwork()
    net = net.to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0,Config.train_number_epochs):
        start = time.time()
        for i, data in enumerate(train_dataloader,0):
            _, __, img0, img1 , label = data
            # img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)

    #         print("In traing loop: {0}".format(img0.shape))
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            end_time = time.time() - start
            if i %30 == 0 :
                print("Epoch number {} iteration number {} Current loss {}\n".format(epoch, i, loss_contrastive.item()))
                with open("train_logs.txt", "a") as f:
                    f.write("Epoch number {} iteration number {} Current loss {} Time taken for this epoch in seconds {}\n"
                            .format(epoch, i, loss_contrastive.item(), end_time))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter,loss_history)

    torch.save(net, "model_")

if __name__ == "__main__":
    main()
    