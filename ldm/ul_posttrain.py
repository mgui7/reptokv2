"""Joint UL post-training initialized from a complete trained RepTok generator."""
import torch
from jutils import load_partial_from_config
from ldm.codec import plain, canonical_state
from ldm.ul import ULRepTok


class ULPosttrainRepTok(ULRepTok):
    validation_metric = 'gfid'
    gradient_log_interval = 50
    extra_log_fields = ULRepTok.extra_log_fields + [
        'train/weighted_prior_loss', 'train/grad_norm_encoder',
        'train/grad_norm_decoder', 'train/grad_norm_prior',
    ]

    def __init__(self, encoder_lr=1e-6, freeze_decoder_time_mlp=False, **kwargs):
        if encoder_lr <= 0:
            raise ValueError('Encoder learning rate must be positive; no frozen-E phase.')
        super().__init__(**kwargs)
        self.encoder_lr = encoder_lr
        self.hyper_parameters['encoder_lr'] = encoder_lr
        self.freeze_decoder_time_mlp = freeze_decoder_time_mlp
        self.hyper_parameters['freeze_decoder_time_mlp'] = freeze_decoder_time_mlp
        if freeze_decoder_time_mlp:
            self.model.net.t_embedder.mlp.requires_grad_(False)
            count = sum(p.numel() for p in self.model.net.t_embedder.mlp.parameters())
            print(f'Frozen decoder time MLP: {count:,} parameters', flush=True)

    def forward(self, batch):
        loss, metrics = super().forward(batch)
        metrics['train/weighted_prior_loss'] = self.prior_weight * metrics['train/token_loss']
        return loss, metrics

    def gradient_metrics(self):
        # DDP has synchronized gradients by this point; measure before clipping.
        values = {}
        for name, net in [('encoder', self.tokenizer), ('decoder', self.model), ('prior', self.prior)]:
            norms = [torch.linalg.vector_norm(p.grad.detach()) for p in net.parameters() if p.grad is not None]
            values[f'train/grad_norm_{name}'] = torch.linalg.vector_norm(torch.stack(norms))
        return values

    def configure_optimizers(self):
        decoder_prior = [p for net in (self.model, self.prior) for p in net.parameters() if p.requires_grad]
        encoder = [p for p in self.tokenizer.parameters() if p.requires_grad]
        if not encoder:
            raise ValueError('UL post-training requires a trainable encoder.')
        optimizer = torch.optim.AdamW([
            dict(params=decoder_prior, lr=self.lr, name='decoder_prior'),
            dict(params=encoder, lr=self.encoder_lr, name='encoder'),
        ], weight_decay=self.weight_decay)
        scheduler = (load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
                     if self.lr_scheduler_cfg else None)
        return optimizer, scheduler


def reptok_initialization(checkpoint, settings):
    """Map the evaluated E/D/G triplet, without carrying stage-2 optimizer state.

    The frozen codec contains the actual decoder used by the trained generator
    (already selected from reconstruction EMA). Seed BOTH train/EMA decoders from
    that codec. Portable architecture configs avoid all external initializer paths.
    """
    base = plain(checkpoint['hyper_parameters'])
    codec = base['codec_config']
    params = plain(settings)
    compile_decoder = params['flow_cfg']['params']['net_cfg']['params'].get('compile', False)
    compile_encoder = params['tokenizer_cfg']['params'].get('compile', False)
    for key in ('flow_cfg', 'tokenizer_cfg', 'first_stage_cfg'):
        params[key] = plain(codec[key])
    params['prior_cfg'] = plain(base['flow_cfg'])
    params['flow_cfg']['params']['net_cfg']['params']['compile'] = compile_decoder
    params['tokenizer_cfg']['params']['compile'] = compile_encoder
    state = canonical_state(checkpoint['state_dict'])
    generator = 'ema_model.' if any(k.startswith('ema_model.') for k in state) else 'model.'
    mapped = {}
    for source, targets in (
        ('codec.flow.', ('model.', 'ema_model.')),
        ('codec.tokenizer.', ('tokenizer.',)),
        ('codec.first_stage.', ('first_stage.',)),
        (generator, ('prior.',)),
    ):
        selected = {key[len(source):]: value for key, value in state.items() if key.startswith(source)}
        if not selected:
            raise ValueError(f'Missing trained component: {source}')
        for target in targets:
            if target == 'ema_model.' and params['ema_rate'] <= 0:
                continue
            mapped.update({target + key: value for key, value in selected.items()})
    decoder = params['flow_cfg']['params']['net_cfg']['params']
    shape = (decoder['in_channels'], decoder['input_size'], decoder['input_size'])
    return dict(state_dict=mapped, hyper_parameters=params,
                latent_shape=shape,
                initialization=dict(source_generator_step=checkpoint['global_step'],
                                    generator_prefix=generator,decoder_prefix='codec.flow.',
                                    optimizer_reset=True))
