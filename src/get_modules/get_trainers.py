from trainers.bidkt_trainer import BidktTrainer
from trainers.bert4kt_plus_trainer import Bert4ktPlusTrainer

def get_trainers(model, optimizer, device, num_q, crit, config):

    #trainer 실행
    if config.model_name == "bidkt":
        trainer = BidktTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "bert4kt_plus":
        trainer = Bert4ktPlusTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    else:
        print("wrong model was choosed..")

    return trainer
