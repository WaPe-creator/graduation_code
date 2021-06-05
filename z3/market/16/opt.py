import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="../Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')
                    
parser.add_argument('--stn_image',
                    type=str,
                    default='0036_c6s1_003851_04.jpg',
                    help='path to the image you want to visit stn (all set)')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model_200.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=200,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=3e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[140, 180],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=8,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    help='the batch size for test')


opt = parser.parse_args()
