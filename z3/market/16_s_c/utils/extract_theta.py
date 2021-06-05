import torch

def extract_theta(model, loader):

    inputs = loader
    input_img = inputs.to('cuda')    
    x = model.backbone(input_img)
    
    xp = model.part2(x)
    xr = model.localization(xp)
    xr = xr.view(-1, 16 * 3 * 1)
    theta = model.fc_loc(xr)
    region_theta = []
    for region in range(6):
        if region == 0:
            theta[:,].add_(-5/6)
            theta_i = model.transform_theta_(theta, region).squeeze(0)
        if region ==1:
            theta[:,].add_(2/6)
            theta_i = model.transform_theta_(theta, region).squeeze(0)
        if region ==2:
            theta[:,].add_(2/6)
            theta_i = model.transform_theta_(theta, region).squeeze(0)
        if region ==3:
            theta[:,].add_(2/6)
            theta_i = model.transform_theta_(theta, region).squeeze(0)
        if region ==4:
            theta[:,].add_(2/6)
            theta_i = model.transform_theta_(theta, region).squeeze(0)
        if region ==5:
            theta[:,].add_(2/6)
            theta_i = model.transform_theta_(theta, region).squeeze(0)
            
        region_theta.append(theta_i)
            
    return (region_theta[0], region_theta[1], region_theta[2], region_theta[3], region_theta[4], region_theta[5])
