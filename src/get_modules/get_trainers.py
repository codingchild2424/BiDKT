from trainers.bidkt_trainer import BidktTrainer
from trainers.bert4kt_plus_trainer import Bert4ktPlusTrainer
from trainers.bert4kt_rasch_trainer import Bert4ktRaschTrainer
from trainers.albert4kt_plus_trainer import ALBert4ktPlusTrainer
from trainers.ma_bert4kt_plus_trainer import MonotonicBert4ktPlusTrainer
from trainers.bcaa_kt_trainer import BcaaKtTrainer
from trainers.nma_bert4kt_dualenc_kr_trainer import NmaBert4ktDualencKrTrainer
from trainers.ma_bert4kt_dualenc_kr_trainer import MaBert4ktDualencKrTrainer
from trainers.bigbird4kt_plus_trainer import Bigbird4ktPlusTrainer
from trainers.bert4kt_plus_time_trainer import Bert4ktPlusTimeTrainer
from trainers.convbert4kt_plus_trainer import ConvBert4ktPlusTrainer
from trainers.monaconvbert4kt_plus_trainer import MonaConvBert4ktPlusTrainer
from trainers.forgetting_monoconvbert4kt_plus_trainer import ForgettingMonoConvBert4ktPlusTrainer
from trainers.monaconvbert4kt_plus_pt_trainer import MonaConvBert4ktPlusPastTrialTrainer
from trainers.monaconvbert4kt_plus_diff_trainer import MonaConvBert4ktPlusDiffTrainer
from trainers.monaconvbert4kt_plus_diff_pt_trainer import MonaConvBert4ktPlusDiffPtTrainer
from trainers.convbert4kt_plus_diff_trainer import ConvBert4ktPlusDiffTrainer
from trainers.monabert4kt_plus_diff_trainer import MonaBert4ktPlusDiffTrainer
from trainers.bert4kt_plus_diff_trainer import Bert4ktPlusDiffTrainer

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
    elif config.model_name == "bert4kt_rasch":
        trainer = Bert4ktRaschTrainer(
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
    elif config.model_name == "albert4kt_plus":
        trainer = ALBert4ktPlusTrainer(
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
    elif config.model_name == "ma_bert4kt_plus":
        trainer = MonotonicBert4ktPlusTrainer(
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
    elif config.model_name == "nma_bert4kt_dualenc_kr":
        trainer = NmaBert4ktDualencKrTrainer(
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
    elif config.model_name == "ma_bert4kt_dualenc_kr":
        trainer = MaBert4ktDualencKrTrainer(
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
    elif config.model_name == "bcaa_kt":
        trainer = BcaaKtTrainer(
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
    elif config.model_name == "bigbird4kt_plus":
        trainer = Bigbird4ktPlusTrainer(
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
    elif config.model_name == "bert4kt_plus_time":
        trainer = Bert4ktPlusTimeTrainer(
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
    elif config.model_name == "convbert4kt_plus":
        trainer = ConvBert4ktPlusTrainer(
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
    elif config.model_name == "monaconvbert4kt_plus":
        trainer = MonaConvBert4ktPlusTrainer(
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
    elif config.model_name == "forgetting_monoconvbert4kt_plus":
        trainer = ForgettingMonoConvBert4ktPlusTrainer(
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
    elif config.model_name == "monaconvbert4kt_plus_pt":
        trainer = MonaConvBert4ktPlusPastTrialTrainer(
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
    elif config.model_name == "monaconvbert4kt_plus_diff":
        trainer = MonaConvBert4ktPlusDiffTrainer(
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
    elif config.model_name == "monaconvbert4kt_plus_diff_pt":
        trainer = MonaConvBert4ktPlusDiffPtTrainer(
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
    elif config.model_name == "convbert4kt_plus_diff":
        trainer = ConvBert4ktPlusDiffTrainer(
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
    elif config.model_name == "monabert4kt_plus_diff":
        trainer = MonaBert4ktPlusDiffTrainer(
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
    elif config.model_name == "bert4kt_plus_diff":
        trainer = Bert4ktPlusDiffTrainer(
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
