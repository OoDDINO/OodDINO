import argparse
from collections import OrderedDict
import torch.optim
import torch
from config.config import config
from dataset.training.cityscapes import Cityscapes
from dataset.validation.fishyscapes import Fishyscapes
from dataset.validation.lost_and_found import LostAndFound
from dataset.validation.road_anomaly import RoadAnomaly
from dataset.validation.anomaly_score import AnomalyDataset
from dataset.validation.segment_me_if_you_can import SegmentMeIfYouCan
from engine.engine import Engine
from model.network import Network
from utils.img_utils import Compose, Normalize, ToTensor
from utils.wandb_upload import *
from utils.logger import *
from engine.evaluator import SlidingEval
# from valid import valid_anomaly, valid_epoch, final_test_inlier
from torch.utils.data import DataLoader
from threshold_Module import ThresholdNet
from utils.pyt_utils import eval_ood_measure
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
from sigmoid import ScoreNormalizer 
from threshold_Module import reverseCompressedSigmoid

def get_anomaly_detector(ckpt_path):
    """
    Get Network Architecture based on arguments provided
    """
    ckpt_name = ckpt_path
    model = Network(config.num_classes)
    state_dict = torch.load(ckpt_name)
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    state_dict = state_dict['model_state'] if 'model_state' in state_dict.keys() else state_dict
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict.keys() else state_dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=True)
    return model


def main(gpu, ngpus_per_node, config, args):
    args.local_rank = gpu
    logger = logging.getLogger("ours")
    logger.propagate = False

    engine = Engine(custom_arg=args, logger=logger,
                    continue_state_object=config.pretrained_weight_path)

    transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])

    # cityscapes_val = Cityscapes(root=config.city_root_path, split="val", transform=transform)
    # cityscapes_test = Cityscapes(root=config.city_root_path, split="test", transform=transform)
    evaluator = SlidingEval(config, device=0 if engine.local_rank < 0 else engine.local_rank)
    fishyscapes_ls = Fishyscapes(split='LostAndFound', root=config.fishy_root_path, transform=transform)
    fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path, transform=transform)
    # segment_me_anomaly = SegmentMeIfYouCan(split='road_anomaly', root=config.segment_me_root_path, transform=transform)
    # segment_me_obstacle = SegmentMeIfYouCan(split='road_obstacle', root=config.segment_me_root_path,
    #                                         transform=transform)
    road_anomaly = RoadAnomaly(root=config.road_anomaly_root_path, transform=transform)
    # lost_and_found = LostAndFound(root=config.lost_and_found_root_path, transform=transform)
    model = get_anomaly_detector(config.rpl_corocl_weight_path)
    vis_tool = Tensorboard(config=config)

    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)
        model.cuda(engine.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[engine.local_rank],
                                                          find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    model.eval()
    """
    # 1). we currently only support single gpu valid for the cityscapes sliding validation, and it 
    # might take long time, feel free to uncomment it. (we'll have to use the sliding eval. to achieve 
      the performance reported in the GitHub. )
    # 2). we follow Meta-OoD to use single scale validation for OoD dataset, for fair comparison.
    """
#     valid_anomaly(model=model, engine=engine, iteration=0, test_set=segment_me_anomaly,
#                   data_name='segment_me_anomaly', my_wandb=vis_tool, logger=logger,
#                   measure_way=config.measure_way)

#     valid_anomaly(model=model, engine=engine, iteration=0, test_set=segment_me_obstacle,
#                   data_name='segment_me_obstacle', my_wandb=vis_tool, logger=logger,
#                   measure_way=config.measure_way)

    # valid_anomaly(model=model, engine=engine, iteration=0, test_set=fishyscapes_static,
    #               data_name='Fishyscapes_static', my_wandb=vis_tool, logger=logger,
    #               measure_way=config.measure_way)

    # valid_anomaly(model=model, engine=engine, iteration=0, test_set=fishyscapes_ls,
    #               data_name='Fishyscapes_ls', my_wandb=vis_tool, logger=logger,
    #               measure_way=config.measure_way)

    anomaly_scores_list, ood_gts_list, bboxes_list = generate_training_data(model=model, engine=engine, iteration=0, test_set=road_anomaly, data_name='road_anomaly',
                  my_wandb=vis_tool, logger=logger, measure_way=config.measure_way)
    # as_score = numpy.array(anomaly_scores_list)
    # lbl_mask = numpy.array(ood_gts_list)
    # # train_id_in = 0, train_id_out = 1
    # roc_auc, prc_auc, fpr = eval_ood_measure( as_score, lbl_mask , 0 , 1)
    # print("roc_auc, prc_auc, fpr--------------", roc_auc, prc_auc, fpr)
    
    
    train_dataset = AnomalyDataset(anomaly_scores_list, ood_gts_list, bboxes_list)
    test_dataset = AnomalyDataset(anomaly_scores_list, ood_gts_list, bboxes_list)
    train_loader = DataLoader( train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader( test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    
    threshold_net = ThresholdNet().cuda()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 或者使用 clip_grad_value_

    optimizer = torch.optim.Adam(threshold_net.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1) 
    
    num_epochs = 1000
    best_fpr = float('inf')  # FPR 越低越好，因此初始值设为无穷大
    best_prc_auc = 0.0
    # output_dir = "threshold_output"
    # best_model_path ="threshold_output"
    for epoch in range(num_epochs):
        threshold_net.train()
        total_loss = 0

        for batch in train_loader:
            # print("batch-------------", batch)
            anomaly_scores = batch['anomaly_score'].cuda()
            labels = batch['ood_gt'].cuda()
            bboxes = batch['bbox']   
            
            batch_size, H, W = anomaly_scores.shape
            # bbox_masks = create_bbox_mask(bboxes, (H, W)).cuda()
            bbox_masks = torch.zeros((batch_size, 1, H, W), dtype=torch.float32).cuda()
            # pred_scores = torch.zeros((batch_size, 1, H, W), dtype=torch.float32).cuda()

            
            for i in range(batch_size):
                bbox_masks[i] = create_bbox_mask(bboxes_list[i], (H, W))
                # max_score = anomaly_scores[i].max()
                # min_score = anomaly_scores[i].min()
                # normalizer = ScoreNormalizer(min_score, max_score)
                # pred_scores[i, 0] = normalizer.normalize(anomaly_scores[i])
                # print(" pred_scores[i, 0]==========",  pred_scores[i, 0])
            # mask = torch.ones_like(labels).cuda()  # 假设没有 bbox 时，mask 全为 1

            # 前向传播
            # print(" pred_scores =====",  pred_scores)
            # print("anomaly_scores.unsqueeze(1), bbox_masks", anomaly_scores.unsqueeze(1).shape, bbox_masks.shape)
            global_thresholds = threshold_net(anomaly_scores.unsqueeze(1), bbox_masks)
            # global_thresholds = threshold_net(pred_scores, bbox_masks)
            
            loss = threshold_net.compute_loss(anomaly_scores.unsqueeze(1), labels.unsqueeze(1), global_thresholds,  bbox_masks)
            # loss = threshold_net.compute_loss(pred_scores, labels.unsqueeze(1), global_thresholds,  bbox_masks)
            print("loss-----------", loss)
            # print("Loss value:", loss)
            # print("Loss requires_grad:", loss.requires_grad)
            # print("Loss grad_fn:", loss.grad_fn)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for name, param in  threshold_net.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad norm: {param.grad.norm()}")
            
        scheduler.step() 
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss:.4f}")
        
        print(f"Starting validation for Epoch {epoch + 1}...")
        thresholds = global_thresholds.mean().item()  # 阈值，示例中使用平均值
        print("thresholds=====1", thresholds)
        # thresholds =normalizer.denormalize( thresholds).cpu().numpy() 
        # print("thresholds=====2", thresholds)
        roc_auc, prc_auc, fpr = valid_anomaly(
            model=threshold_net,
            engine=None,  # 你的分布式训练配置，如果没有，传 None
            test_set=test_loader,  # 验证集或测试集
            data_name="road_anomaly",  # 数据集名称
            iteration=epoch,
            threshold=thresholds,  # 传入计算出的阈值
            logger=None  # 日志记录器
        )
        output_dir = "threshold_output"  # 输出路径
        best_model_path = os.path.join(output_dir, "best_model.pth")
        best_info_path = os.path.join(output_dir, "best_model_info.json")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 计算并打印 FPR, PRC-AUC 和 ROC-AUC
        print(f"roc_auc, prc_auc, fpr-------------- {roc_auc:.4f}, {prc_auc:.4f}, {fpr:.4f}")

        # 判断是否保存最佳模型
        if fpr < best_fpr or prc_auc > best_prc_auc:
            best_fpr = fpr
            best_prc_auc = prc_auc

            # 删除旧的最佳模型文件（如果存在）
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
                print("Deleted previous best model weights.")

            # 保存新的最佳模型
            torch.save(threshold_net.state_dict(), best_model_path)
            print(f"Saved best model at Epoch {epoch + 1} with FPR: {fpr:.4f}, PRC-AUC: {prc_auc:.4f}")

            # 保存最佳模型的epoch、fpr和prc_auc到JSON文件
            best_info = {
                "epoch": epoch + 1,
                "fpr": fpr,
                "prc_auc": prc_auc
            }

            # 检查文件保存路径
            print(f"Saving best model info to {best_info_path}...")

            with open(best_info_path, 'w') as f:
                json.dump(best_info, f, indent=4)
                print(f"Best model info saved to {best_info_path}.")
            

    
def valid_anomaly(model, engine, test_set, data_name=None, iteration=None, threshold=0.0, my_wandb=None, logger=None, 
                  upload_img_num=4, measure_way="energy"):
    if engine is not None and engine.local_rank <= 0:
        logger.info(f"Validating {data_name} dataset with {measure_way} ...")

    model.eval()
    anomaly_score_list = []
    ood_gts_list = []
    bboxes_list = []
    Pred = reverseCompressedSigmoid()
    threshold = torch.tensor(threshold)
    
    threshold = Pred(threshold)
    threshold = threshold.item()
    # print("threshold---==-=-=", threshold)
    
    with torch.no_grad():
        for batch in test_set:  # 假设 batch 是一个包含多张图像数据的字典
            anomaly_scores = batch['anomaly_score']  # (B, H, W) 异常分数
            ood_gts = batch['ood_gt']  # (B, H, W) 标签
            bboxes = batch['bbox']  # 每张图像的边界框列表，格式 [[(x1, y1, x2, y2), ...], ...]

            for i in range(len(anomaly_scores)):  # 遍历每张图像
                visual = anomaly_scores[i].detach().cpu().numpy()  # 当前图像的异常分数 (H, W)
                ood_gt = ood_gts[i].detach().cpu().numpy()  # 当前图像的标签 (H, W)
                image_bboxes = bboxes[i]  # 当前图像的边界框 [(x1, y1, x2, y2), ...]

                min_value = visual.min()  # 图像异常分数的最小值
                mask = np.full_like(visual, min_value)  # 初始化 mask，默认值为最小异常分数

                # 遍历边界框，框内异常分数保留不变
                for bbox in image_bboxes:
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        mask[y1:y2, x1:x2] = visual[y1:y2, x1:x2]  # 框内分数保持不变
                    else:
                        logger.warning(f"Unexpected bbox format: {bbox}")

                # 框外分数大于阈值的保留，不符合的设为最小值
                outside_bbox = (mask == min_value)
                mask[outside_bbox & (visual > threshold)] = visual[outside_bbox & (visual > threshold)]

                # 将处理后的异常分数和对应的标签存入列表
                anomaly_score_list.append(mask)
                ood_gts_list.append(ood_gt)
                bboxes_list.append(image_bboxes)

    # 转换为 NumPy 数组以便后续计算
    anomaly_scores = np.array(anomaly_score_list)
    ood_gts = np.array(ood_gts_list)

    # 多 GPU 同步（如果有多 GPU）
    if engine is not None and engine.gpus > 1:
        anomaly_scores, ood_gts = all_gather_samples(
            x_=anomaly_scores, y_=ood_gts, local_rank=engine.local_rank
        )

    # 计算评价指标
    if engine is None or engine.local_rank <= 0:
        roc_auc, prc_auc, fpr = eval_ood_measure(
            anomaly_scores, ood_gts, 0 , 1
        )
        # logger.info(f"ROC AUC: {roc_auc:.4f}, PRC AUC: {prc_auc:.4f}, FPR: {fpr:.4f}")
        return roc_auc, prc_auc,fpr



    
def custom_collate_fn(batch):
    anomaly_scores = torch.stack([item['anomaly_score'] for item in batch])
    ood_gts = torch.stack([item['ood_gt'] for item in batch])
    bboxes = [item['bbox'] for item in batch]  # 保留为列表形式
    return {
        "anomaly_score": anomaly_scores,
        "ood_gt": ood_gts,
        "bbox": bboxes
    }

def create_bbox_mask(bboxes, img_size):
    """
    创建框掩码，框外为1，框内为0。
    参数：
        bboxes: List[List[int]], 每个框的信息 [x_min, y_min, x_max, y_max]
        img_size: Tuple[int, int], 图像大小 (H, W)
    返回：
        mask: torch.Tensor, 框掩码 (1, H, W)
    """
    mask = torch.ones(img_size, dtype=torch.float32)  # 初始全为1
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        mask[y_min:y_max, x_min:x_max] = 0  # 框内置为0
    return mask.unsqueeze(0)  # 添加批次维度

    
def generate_training_data(model, engine, test_set, data_name=None, iteration=None, my_wandb=None, logger=None,upload_img_num=4, measure_way="energy"):
    """
    从 test_set 中提取 anomaly_scores、ood_gts、bboxes 等数据。

    参数:
        model: PyTorch 模型，用于计算异常分数。
        test_set: 数据集，提供图像和标签。
        coco_json_path: JSON 文件路径，存储检测框。
        measure_way: 计算异常分数的方式 ("energy" 或其他方式)。

    返回:
        anomaly_scores_list: 所有图像的异常分数列表。
        ood_gts_list: 所有图像的 GT 标注列表。
        bboxes_list: 所有图像的检测框列表。
    """
    anomaly_scores_list = []
    ood_gts_list = []
    bboxes_list = []
    coco_json_path = 'vis_data/roadanomaly_vis/preds'
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for idx in range(len(test_set)):
            img, label, img_path = test_set[idx]
            img_name = os.path.basename(img_path)
            bboxes = get_bbox_v1(img_name, coco_json_path)

            img = img.cuda(non_blocking=True)
            inlier_logits, logits, _ = model(img)

            anomaly_score = compute_anomaly_score(logits, mode=measure_way).cpu()
            anomaly_scores_list.append(anomaly_score.numpy())
            ood_gts_list.append(label.numpy())
            bboxes_list.append(bboxes)

    return anomaly_scores_list, ood_gts_list, bboxes_list

def compute_anomaly_score(score, mode='energy'):
   
    score = score.squeeze()[:19]
    # print("score----------", score)
    if mode == 'energy':
        anomaly_score = -(1. * torch.logsumexp(score, dim=0))
    elif mode == 'entropy':
        prob = torch.softmax(score, dim=0)
        anomaly_score = -torch.sum(prob * torch.log(prob), dim=0) / torch.log(torch.tensor(19.))
    else:
        raise NotImplementedError

    # regular gaussian smoothing
    anomaly_score = anomaly_score.unsqueeze(0)
    anomaly_score = torchvision.transforms.GaussianBlur(7, sigma=1)(anomaly_score)
    anomaly_score = anomaly_score.squeeze(0)
    return anomaly_score

def get_bbox_v1(img_name, json_dir):
    """
    从对应的 JSON 文件中获取 scores 大于 0.8 的 bbox 信息 (x1, y1, x2, y2 格式)。
    
    参数：
    - img_name (str): 图像文件名，例如 "image_0.png"。
    - json_dir (str): JSON 文件所在的目录路径。

    返回：
    - List[List[int]]: 筛选出的 bbox 列表，格式为 [[x1, y1, x2, y2], ...]。
    """
    # 构造 JSON 文件名
    json_name = os.path.splitext(img_name)[0] + ".json"
    json_path = os.path.join(json_dir, json_name)

    # 检查 JSON 文件是否存在
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found for image: {img_name}")

    # 加载 JSON 数据
    with open(json_path, "r") as f:
        data = json.load(f)

    # 提取 keys
    labels = data.get("labels", [])
    scores = data.get("scores", [])
    bboxes = data.get("bboxes", [])

    # 检查数据完整性
    if not (len(labels) == len(scores) == len(bboxes)):
        raise ValueError(f"Data inconsistency in JSON file: {json_path}")

    # 筛选 scores > 0.8 的 bbox，并转换为 x1, y1, x2, y2 格式
    filtered_bboxes = []
    for score, bbox in zip(scores, bboxes):
        if score > 0.9:
            # 假设 bbox 格式为 [x1, y1, x2, y2]
            filtered_bboxes.append([int(coord) for coord in bbox])

    return filtered_bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Segmentation')
    parser.add_argument('--gpus', default=1,
                        type=int,
                        help="gpus in use")
    parser.add_argument("--ddp", action="store_true",
                        help="distributed data parallel training or not;"
                             "MUST SPECIFIED")
    parser.add_argument('-l', '--local_rank', default=-1,
                        type=int,
                        help="distributed or not")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int,
                        help="distributed or not")

    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus

    # we enforce the flag of ddp if gpus >= 2;
    args.ddp = True if args.world_size > 1 else False
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, config, args))
