import numpy as np
import datetime

import torch
from get_modules.get_loaders import get_loaders
from get_modules.get_models import get_models
from get_modules.get_trainers import get_trainers
from utils import get_average_scores, get_optimizers, get_crits, recorder

from define_argparser import define_argparser

def main(config, train_loader=None, test_loader=None, num_q=None, num_r=None):
    #0. device 선언
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    #1. 데이터 받아오기
    if config.fivefold == True:
        train_loader = train_loader
        test_loader = test_loader
        num_q = num_q
        num_r = num_r
    else:
        train_loader, test_loader, num_q, num_r = get_loaders(config)

    #2. model 선택
    model = get_models(num_q, num_r, device, config)
    
    #3. optimizer 선택
    optimizer = get_optimizers(model, config)
    
    #4. criterion 선택
    crit = get_crits(config)
    
    #5. trainer 선택
    trainer = get_trainers(model, optimizer, device, num_q, crit, config)

    #6. 훈련 및 score 계산
    y_true_record, y_score_record, \
        train_auc_scores, test_auc_scores, \
        highest_auc_score = trainer.train(train_loader, test_loader)

    today = datetime.datetime.today()
    record_time = str(today.month) + "_" + str(today.day) + "_" + str(today.hour) + "_" + str(today.minute)

    #7. model 기록 저장 위치
    model_path = '../model_records/' + str(round(highest_auc_score, 6)) + "_" + record_time + "_" + config.model_fn

    #8. model 기록
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    return train_auc_scores, test_auc_scores, highest_auc_score, record_time

#fivefold main
if __name__ == "__main__":
    config = define_argparser() #define_argparser를 불러옴

    if config.fivefold == True:

        highest_auc_scores = []
        train_auc_scores_list = []
        test_auc_scores_list = []

        for idx in range(5):
            train_loader, test_loader, num_q, num_r = get_loaders(config, idx)
            train_auc_scores, test_auc_scores, highest_auc_score, record_time = main(config, train_loader, test_loader, num_q, num_r)
            highest_auc_scores.append(highest_auc_score)
            train_auc_scores_list.append(train_auc_scores)
            test_auc_scores_list.append(test_auc_scores)

        highest_auc_scores = sum(highest_auc_scores)/5
        train_auc_scores = get_average_scores(train_auc_scores_list)
        test_auc_scores = get_average_scores(test_auc_scores_list)

        recorder(train_auc_scores, test_auc_scores, highest_auc_scores, record_time, config)        
    else:
        train_auc_scores, test_auc_scores, highest_auc_score, record_time = main(config)
        recorder(train_auc_scores, test_auc_scores, highest_auc_score, record_time, config)
    


# <main>
# if __name__ == "__main__":
#     config = define_argparser() #define_argparser를 불러옴

#     train_auc_scores, test_auc_scores, highest_auc_score, record_time = main(config)
#     # 기록, utils에 있음
#     recorder(train_auc_scores, test_auc_scores, highest_auc_score, record_time, config)

    