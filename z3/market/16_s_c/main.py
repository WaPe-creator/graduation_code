import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import Model1
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.extract_theta import extract_theta
from utils.metrics import mean_ap, cmc, re_ranking

import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        #self.model = torch.nn.DataParallel(model).cuda()

        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')
        
    def vis_stn(self):
        
        self.model.eval()
        w = 64
        h = 128
        os.makedirs('stn', exist_ok=True)
        # os.makedirs('stn/s1', exist_ok=True)
        # os.makedirs('stn/s2', exist_ok=True)
        
        img = cv2.imread(opt.stn_image, 1)

        theta1, theta2, theta3, theta4, theta5, theta6 = extract_theta(model, torch.unsqueeze(data.stn_image, 0))
        # print(theta1)
        
        # for i in range(6):
            
        #    x1 = w/2*(1+theta[i][0,2]-theta[i][0,0])
        #    y1 = h/2*(1+theta[i][1,2]-theta[i][1,1])
        #    cv2.rectangle(img, (x1, y1), (x1+w*theta[i][0,0]-1, y1+h*theta[i][1,1]-1), (0,0,255), 2)
        
        x1 = w/2*(1+theta1[0,2]-theta1[0,0])
        y1 = h/2*(1+theta1[1,2]-theta1[1,1])
        cv2.rectangle(img, (x1, y1), (x1+w*theta1[0,0]-1, y1+h*theta1[1,1]-1), (0,0,255), 1)
        
        x2 = w/2*(1+theta2[0,2]-theta2[0,0])
        y2 = h/2*(1+theta2[1,2]-theta2[1,1])
        cv2.rectangle(img, (x2, y2), (x2+w*theta2[0,0]-1, y2+h*theta2[1,1]-1), (0,255,255), 1)
        
        x3 = w/2*(1+theta3[0,2]-theta3[0,0])
        y3 = h/2*(1+theta3[1,2]-theta3[1,1])
        cv2.rectangle(img, (x3, y3), (x3+w*theta3[0,0]-1, y3+h*theta3[1,1]-1), (255,0,0), 1)
        
        x4 = w/2*(1+theta4[0,2]-theta4[0,0])
        y4 = h/2*(1+theta4[1,2]-theta4[1,1])
        cv2.rectangle(img, (x4, y4), (x4+w*theta4[0,0]-1, y4+h*theta4[1,1]-1), (0,0,255), 1)
        
        x5 = w/2*(1+theta5[0,2]-theta5[0,0])
        y5 = h/2*(1+theta5[1,2]-theta5[1,1])
        cv2.rectangle(img, (x5, y5), (x5+w*theta5[0,0]-1, y5+h*theta5[1,1]-1), (0,255,255), 1)
        
        x6 = w/2*(1+theta6[0,2]-theta6[0,0])
        y6 = h/2*(1+theta6[1,2]-theta6[1,1])
        cv2.rectangle(img, (x6, y6), (x6+w*theta6[0,0]-1, y6+h*theta6[1,1]-1), (255,0,0), 1)
        
        cv2.imwrite("stn/stn_{}.png".format(opt.stn_image.split('/')[-1]), img)
            
        print('result saved to show_stn.png')


if __name__ == '__main__':

    data = Data()
    model = Model1()
    loss = Loss()
    main = Main(model, loss, data)

    if opt.mode == 'train':

        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            main.train()
            if epoch % 50 == 0:
                print('\nstart evaluate')
                main.evaluate()
                os.makedirs('weights', exist_ok=True)
                torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
        
    if opt.mode == 'stn':
        print('visualize stn')
        model.load_state_dict(torch.load(opt.weight))
        main.vis_stn()
