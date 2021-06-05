import torch

def extract_theta(model, loader, num):

    inputs = loader
    input_img = inputs.to('cuda')    
    x = model.backbone(input_img)
    
    if num == 2: 
        xp2 = model.part2(x)
        xr2 = model.localization_2(xp2)
        xr2 = xr2.view(-1, 16 * 3 * 1)
        theta = model.fc_loc_2(xr2)
        region_theta = []
        for region in range(2):
            if region == 0:
                theta[:,].add_(-1/2)
                theta2_i = model.transform_theta_2(theta, region).squeeze(0)
            if region ==1:
                theta[:,].add_(1)
                theta2_i = model.transform_theta_2(theta, region).squeeze(0)
            
            region_theta.append(theta2_i)
            
        return region_theta[0], region_theta[1]

    if num == 3: 
        xp3 = model.part3(x)
        xr3 = model.localization_3(xp3)
        xr3 = xr3.view(-1, 16 * 3 * 1)
        # print(xr3)
        theta = model.fc_loc_3(xr3)
        # print(theta)
        region_theta = []
        for region in range(3):
            if region == 0:
                theta[:,].add_(-2/3)
                theta3_i = model.transform_theta_3(theta, region).squeeze(0)
            if region ==1:
                theta[:,].add_(2/3)
                theta3_i = model.transform_theta_3(theta, region).squeeze(0)
            if region == 2:
                theta[:,].add_(2/3)
                theta3_i = model.transform_theta_3(theta, region).squeeze(0)
            
            region_theta.append(theta3_i)
            
        return region_theta[0], region_theta[1], region_theta[2]
