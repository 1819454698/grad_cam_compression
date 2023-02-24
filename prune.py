import torch
from models.vgg import vgg19_bn,vgg16_bn
from test import base_architecture_to_features
import numpy as np
from utils.dataset import load_data
from utils.train import eval,get_criterion
from utils.utils import Conv2d_Attri
from utils.models import is_leaf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model=base_architecture_to_features['resnet18']()
model.load_state_dict(torch.load("/home/explainable/src/Grad-cam-compress/compressed_model/resnet18/epoch: 588 81.1765.pth"))
model=model.cuda()
to_cat_mask=[]
radio=0.32
for name,param in model.named_parameters():
    if param.dim()==4:
        print(name)
        to_cat_mask.append(param.view(param.size(0),-1).max(dim=-1).values)
all_result=torch.cat(to_cat_mask)

nz=int(all_result.shape[0]*radio)
print(all_result.shape)
print(nz)
top_values, _ = torch.topk(torch.abs(all_result), nz,sorted=True)
thresh=top_values[-1]
index=0
front=0
mask=[]
def flops_resnet(model,input,radio):
    result=[]
    total=0
    output=model(input)
    for name,module in model._modules.items():
        if is_leaf(module):
            if isinstance(module,Conv2d_Attri):
                result.append(module.flop)
        else:
            for name,module1 in module._modules.items():
                if is_leaf(module1):
                    if isinstance(module1,Conv2d_Attri):
                        result.append(module1.flop)
                else:
                    for name,module2 in module1._modules.items():
                        if is_leaf(module2):
                            if isinstance(module2,Conv2d_Attri):
                                result.append(module2.flop)
                        else:
                            for name,module3 in module2._modules.items():
                                if isinstance(module3,Conv2d_Attri):
                                    result.append(module3.flop)    
    for i in range(len(result)):
        total+=result[i]*radio[i]*radio[i+1]
    return total    
def flops(model,input,radio):
    result=[]
    total=0
    output=model(input)
    for name,module in model.features._modules.items():
        if isinstance(module, Conv2d_Attri):
            result.append(module.flop)
    for i in range(len(result)):
        total+=result[i]*radio[i]*radio[i+1]
    return total
radio_list=[]
radio_list.append(1)
print(model)
for name,param in model.named_parameters():
    
    if param.dim()==4:
        print(name)
        if front==0:
            front=1
        elif 'downsample' not in name:
            for j in range(mask.shape[0]):
                #print(name)
                param.data[:,j,:,:]=mask[j]*param.data[:,j,:,:]
        mask= (torch.abs(to_cat_mask[index])>=thresh).type(torch.cuda.FloatTensor)
        #print(mask)
        radio_list.append(float((mask!=0).sum())/float(mask.shape[0]))
        #print(mask)
        for j in range(mask.shape[0]):
            
            param.data[j,:,:,:] = mask[j] * param.data[j,:,:,:]
        index+=1 
    else:
        if name!='fc.bias':
            param.data*=mask

sum=0.0
sum_nozero=0.0 
for name,param in model.named_parameters():
    #print(param)
    sum+=param.numel()
    sum_nozero+=(param!=0).sum()
    
print(sum)
print(sum_nozero)
print('radio:',sum_nozero/sum)
radio_list[7]=1
radio_list[12]=1
radio_list[17]=1
flops_list=flops_resnet(model,input=torch.randn(1,3,224,224).cuda(),radio=radio_list)
print(flops_list)
print(radio_list)
#print(radio)
'''
train_dataloader,test_dataloader=load_data(32,32)
criterion=get_criterion()


acc_avg,acc5_avg,loss_avg=eval(model,test_dataloader,criterion)
print(acc_avg)
'''
#print(thresh)

'''
for input,label,input_se in test_dataloader:
    input=input.cuda()
    for name,module in model.features._modules.items():
        input=module(input)
        print(input)
    break
'''
