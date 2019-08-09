import cmpnn
import data
from utils import str2bool
import argparse
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training opts')
    parser.add_argument(
        "--with_residual",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Activate residual link')
    parser.add_argument(
        "--with_global_pool",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Activate global pooling')
    parser.add_argument(
        "--min_inliers",
        type=int,
        const=True,
        nargs='?',
        default=30,
        help="number of minimium inliers")
    parser.add_argument(
        "--max_inliers",
        type=int,
        const=True,
        nargs='?',
        default=60,
        help="number of minimium inliers")
    parser.add_argument(
        "--max_outliers",
        type=int,
        const=True,
        nargs='?',
        default=20,
        help="number of maximium inliers")
    parser.add_argument(
        "--lr",
        type=float,
        const=True,
        nargs='?',
        default=1e-3,
        help="learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--epoches", type=int, default=9, help="number of epoches")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    batch_size = args.batch_size

    print(args)

    writer = SummaryWriter()

    dataset = data.RandomGraphDataset(args.min_inliers, args.max_inliers, 0,
                                      args.max_outliers)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    def ce_loss_logits(target, res):
        return -torch.sum(target * res) / torch.sum(target)

    def accurance(target, res):
        return target.argmax(dim=2).eq(res.argmax(dim=2)).sum() / target.sum()

    model = cmpnn.graph_matching.feature_network(
        with_residual=args.with_residual, with_global=args.with_global_pool)
    model.cuda()

    model_name = 'matching_res_{}_gp_{}'.format(args.with_residual,
                                                args.with_global_pool)
    # In[5]:

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # In[ ]:
    icnt = 0
    loss_func = torch.nn.functional.cross_entropy

    def acc_func(output, target):
        toutput = output.argmax(dim=2)
        correct = (target.eq(toutput.long())).sum()
        all_cnt = (target >= 0).long().sum()
        return correct.cpu().item() * 1.0 / all_cnt.cpu().item()

    if os.path.exists('{}_snap.pt'.format(model_name)):
        checkpoint = torch.load('{}_snap.pt'.format(model_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        icnt = checkpoint['icnt']
        print("Model loaded")

    for epoch in tqdm(range(args.epoches)):
        for ibatch, data_batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            pt1, nn_idx1, pt2, nn_idx2, mask, gt = data_batch
            pt1, nn_idx1, pt2, nn_idx2, mask, gt = pt1.cuda(), nn_idx1.cuda(
            ), pt2.cuda(), nn_idx2.cuda(), mask.cuda(), gt.cuda()

            feature1 = model(pt1.permute(0, 2, 1), nn_idx1, mask)
            feature2 = model(pt2.permute(0, 2, 1), nn_idx2, mask)

            sim = torch.bmm(feature1.permute(0, 2, 1), feature2)
            loss = loss_func(sim.view(-1, sim.shape[-1]), gt.view(-1))

            with torch.no_grad():
                acc = acc_func(sim, gt)
            print('epoch={}, ibatch = {}, loss = {}, acc={}'.format(
                epoch, ibatch, loss.item(), acc))
            writer.add_scalar('train/loss', loss.item(), icnt)
            writer.add_scalar('train/acc', acc, icnt)

            icnt += 1
            loss.backward()
            optimizer.step()
            if icnt % 100 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'icnt': icnt
                }, '{}_snap.pt'.format(model_name))

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'icnt': icnt,
        }, '{}_epoch_{}.pt'.format(model_name, epoch))

    writer.close()


if __name__ == '__main__':
    main()
