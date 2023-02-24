import torch
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.train import base_train,attri_train
from models.vgg import vgg19_bn,vgg16_bn
from models.resnet import resnet18,resnet50
from base_config import get_baseconfig_by_epoch
from utils.dataset import load_data 
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg
from utils.train import eval,get_criterion
import numpy as np
from utils.utils import Conv2d_Attri
from tensorflow.keras.preprocessing import image
def get_num_gen(gen):
    return sum(1 for x in gen)
def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name
def is_leaf(model):
    return get_num_gen(model.children()) == 0

base_architecture_to_features = {
                                 'vgg_16':vgg16_bn,
                                 'vgg_19': vgg19_bn,
                                 'resnet18':resnet18,
                                 'resnet50':resnet50}
if __name__ == "__main__":
    model=base_architecture_to_features['resnet50']()
    model.load_state_dict(torch.load("/home/explainable/src/Grad-cam-compress/base_model/resnet50/80.0980.pth"))
    #print(model)
    img_path="/home/explainable/src/dataset/segmin/Oxford 102 Flowers/data/oxford-102-flowers/oxford-102-flowers/jpg/image_08178.jpg"
    img=image.load_img(img_path,target_size=(224,224))
    img_tensor=image.img_to_array(img)
    img_tensor=np.expand_dims(img_tensor,axis=0)
    img_tensor/=255.
    img_tensor_torch=torch.from_numpy(img_tensor).permute(0,3,1,2).cuda()
    X=img_tensor_torch
    
    model=model.cuda()
    print(model)
    class FeatureExtrator():
        def __init__(self,model):
            self.model=model
            self.gradients=[]
        def save_gradients(self,grad):
            self.gradients.append(grad)
        def get_gradients(self):
            return self.gradients
        ###############################################################
        def __call__(self,X):
            activation=[]
            for name1,module1 in self.model._modules.items():
                if is_leaf(module1):
                    if isinstance(module1, Conv2d_Attri):
                        X=module1(X)
                        X.register_hook(self.save_gradients)
                        activation.append(X)
                        
                        continue
                    elif 'fc' in name1:
                        X=X.view(X.shape[0],-1)
                    X=module1(X)
                else:
                    for name2,module2 in module1._modules.items():
                        

                        if is_leaf(module2):
                            if isinstance(module2,torch.nn.Linear):
                                X=X.view(X.shape[0],-1)
                            X=module2(X)
                            if isinstance(module2, Conv2d_Attri):
                                
                                X.register_hook(self.save_gradients)
                                activation.append(X)
                                
                                continue
                        else:
                            identity=X
                            
                            for name3,module3 in module2._modules.items():
                                
                                if is_leaf(module3):
                                    X=module3(X)
                                    if isinstance(module3, Conv2d_Attri):
                                        #print(name1+name2+name3)
                                        
                                        X.register_hook(self.save_gradients)
                                        activation.append(X)
                                        
                                        continue
                                else:
                                    #X3=X1.copy()
                                    for name4,module4 in module3._modules.items():
                                        print(name1+name2+name3+name4)
                                        if is_leaf(module4):
                                            identity=module4(identity)
                                            if isinstance(module4, Conv2d_Attri):

                                                identity.register_hook(self.save_gradients)
                                                activation.append(identity)
                                                
                            X=X+identity
                            X=torch.nn.ReLU(inplace=True)(X)
                
            return X,activation
    model.eval()
    extrator=FeatureExtrator(model)
    output,activation=extrator(X)
    one_hot=output.max()
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    gradients=extrator.get_gradients()
    #print(len(gradients))
    #print(gradients)
    
    

  
    print(output)
    output=model(X)
    print(output)
    #print(len(gradients))
    #print(gradients)
    #print(X)
    #print(model)
              
    #for name,param in model.named_parameters():
    #    print(name)
    #for name,param in torch.load("/home/explainable/src/Grad-cam-compress/base_model/vgg16/92.0588.pth").items():
    #    print(name)
    '''    
    lena = mpimg.imread("/home/explainable/src/Grad-cam-compress/only_mask_1.jpg")
    img=mpimg.imread("/home/explainable/src/Grad-cam-compress/test_1.jpg")
    c=img.copy()
    mask=lena.sum(axis=(2))
    mask=mask.astype(np.int64)
    tensor=torch.tensor(mask)
    tensor=tensor.view(-1)
    top_values, _ = torch.topk(tensor, 224*30)
    print(top_values[-1])
    #c=lena.copy()
    mask=(mask>top_values[-1].item())
    for i in range(3):
        c[:,:,i]*=mask
    plt.imshow(c) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()
    #model=base_architecture_to_features['resnet18']()
    #model.load_state_dict(torch.load("/home/explainable/src/Grad-cam-compress/base_model/resnet18/80.9804.pth"))
    #print(model)
    #model=model.cuda()
    '''
    '''
    for name,param in model.named_parameters():
        print(name)

    train_dataloader,test_dataloader=load_data(32,32)
    criterion=get_criterion()
    acc,acc5,loss=eval(model,test_dataloader,criterion)
    print(acc,acc5,loss)

    '''
    '''
    for i,(name,param )in enumerate(model.named_parameters()):
        print(name)
        print(param)
    '''
    '''
    a=torch.zeros(5)
    b=(a[1:3]>0)
    print(b)
    '''
        


