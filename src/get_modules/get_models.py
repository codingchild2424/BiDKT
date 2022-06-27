from models.bidkt import Bidkt

def get_models(num_q, num_r, device, config):

    if config.model_name == "bidkt":
        model = Bidkt(
            num_q=num_q,
            num_r=num_r,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p,
        ).to(device)
    else:
        print("Wrong model_name was used...")

    return model