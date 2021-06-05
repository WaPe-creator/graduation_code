import torch
import numpy as np

class GCAM():
    def __init__(self, model):
        self.gradients = []
        self.model = model

    # the operation of register_hook()
    def save_grad(self, grad):
        self.gradients.append(grad)

    def make_cam(self, one_hot, features, output):

        output = output*one_hot
        #output = torch.sum(output, 1)
        output = torch.sum(output)

        self.model.zero_grad()
        output.backward()

        grads = self.gradients[-1].cpu().data.numpy()
        weights = np.mean(grads, axis = (2, 3), keepdims=True)
                
        target = features[-1].cpu().data.numpy()

        cam = weights * target
        cam = np.sum(cam,axis=1)
        cam = np.maximum(cam, 0)
        
        return cam


    def __call__(self, x, one_hot, branch):
        self.gradients = []
        features = []
        
        if branch == 'nl':
            x = self.model.backbone(x)
            nl_x, sp_x = self.model.msa(x)
            nl_x.register_hook(self.save_grad)
            features.append(nl_x)
    
            x = self.model.part_nl(nl_x)
            p = self.model.avgpool_nl(x)
            l = self.model.fc_nl(p.squeeze(dim=3).squeeze(dim=2))
    
            cam = self.make_cam(one_hot, features, l)
            
            return cam
            
        if branch == 'sp':
            x = self.model.backbone(x)
            nl_x, sp_x = self.model.msa(x)
            sp_x.register_hook(self.save_grad)
            features.append(sp_x)
    
            x = self.model.part_sp(sp_x)
            p = self.model.avgpool_sp(x)
            l = self.model.fc_sp(p.squeeze(dim=3).squeeze(dim=2))
    
            cam = self.make_cam(one_hot, features, l)
            
            return cam


def get_cam(model, inputs, one_hot):

    gcam = GCAM(model)
    
    cache = []
    branch = ['nl', 'sp']
    
    for i in range(2):
        cam = gcam(inputs, one_hot, branch[i])
        cache.append(cam)
        
    return cache