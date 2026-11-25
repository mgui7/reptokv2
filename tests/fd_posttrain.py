"""Clean-FD routing, corruption, trained initialization and native DDP resume."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import model_params, TinyMetricTracker
from ldm.codec import plain
from ldm.ul import ULRepTok
from ldm.fd_posttrain import FDPosttrainRepTok, clean_fd_initialization
from ldm.train_utils import save_checkpoint, load_checkpoint, train_batch


class SelectionFixture(FDPosttrainRepTok):
    """Run the genuine validation paths with lightweight image metric fixtures."""
    gradient_log_interval = 1
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('validation_calls', torch.tensor(0))

    def validate(self, *args, **kwargs):
        import ldm.metrics
        import ldm.generation
        class TinyGeneration(TinyMetricTracker):
            def aggregate(self):
                return {'gfid': super().aggregate()['fid']}
        original = ldm.metrics.ImageMetricTracker, ldm.generation.GenerationMetrics
        ldm.metrics.ImageMetricTracker, ldm.generation.GenerationMetrics = TinyMetricTracker, TinyGeneration
        updates, moments = self.fd.updates.clone(), self.fd.seconds.clone()
        try:
            values, panels = super().validate(*args, **kwargs)
        finally:
            ldm.metrics.ImageMetricTracker, ldm.generation.GenerationMetrics = original
        assert torch.equal(updates, self.fd.updates) and torch.equal(moments, self.fd.seconds)
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in values.values())
        values['gfid'] = [3., 1., 2.][min(int(self.validation_calls), 2)]
        self.validation_calls.add_(1)
        return values, panels


def main():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    base = plain(model_params())
    params = {k:base[k] for k in ['flow_cfg','tokenizer_cfg','first_stage_cfg','lr_scheduler_cfg']}
    params.update(lr=1e-5, ema_rate=.9, token_steps=1, decoder_steps=3, n_images_to_vis=2)
    params['prior_cfg'] = {'target':'ldm.generation.TokenFlow','params':{
        'net_cfg':{'target':'ldm.models.transformer.token_dit.OneDimensionalTransformer','params':{
            'num_channels':1,'in_channels':32,'hidden_size':32,'depth':2,'num_heads':4,
            'num_classes':3,'class_dropout_prob':.1,'n_registers':1,'split_channels':1}}}}
    if '--xpred' in sys.argv:
        params['prior_cfg']['params'].update(prediction_type='x', t_eps=.05)
    donor = ULRepTok(**params)
    with torch.no_grad():
        for net in (donor.model, donor.prior):
            for p in net.parameters():
                if p.requires_grad: p.add_(torch.randn_like(p)*.03)
    checkpoint = dict(global_step=100000,hyper_parameters={
        'flow_cfg':donor.hyper_parameters['prior_cfg'],
        'codec_config':{k:donor.hyper_parameters[k] for k in ['flow_cfg','tokenizer_cfg','first_stage_cfg']}},state_dict={})
    for net,prefix in [(donor.model,'codec.flow.'),(donor.tokenizer,'codec.tokenizer.'),
                       (donor.first_stage,'codec.first_stage.'),(donor.prior,'model.')]:
        checkpoint['state_dict'].update({prefix+k:v for k,v in net.state_dict().items()})
    settings=dict(params,fd_weight=.1,fd_decay=.99,fd_jitter=1e-5,fd_normalization_epsilon=1e-6,
                  encoder_lr=1e-6,freeze_decoder_time_mlp=True,decoder_noise_reference_std=1.3,
                  decoder_signal_scale=.8,decoder_noise_scale=.6,decoder_clean_probability=.2)
    mapped=clean_fd_initialization(checkpoint,settings)
    module=FDPosttrainRepTok(**mapped['hyper_parameters'])
    module.load_state_dict(mapped['state_dict'],strict=True)
    assert tuple(mapped['latent_shape'])==(4,4,4) and int(module.fd.updates)==0
    assert 'optimizer_states' not in mapped
    for old,new in [(donor.model,module.model),(donor.model,module.ema_model),
                    (donor.tokenizer,module.tokenizer),(donor.prior,module.prior)]:
        for key,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
    # Exact formula and derivative, including a genuinely untouched clean path.
    z=torch.randn(10000,1,32,requires_grad=True); eps=torch.randn_like(z)
    clean=torch.rand(len(z))<.2
    assert .18<float(clean.float().mean())<.22
    out=module.corrupt_tokens(z,eps,clean)
    torch.testing.assert_close(out[clean],z[clean],rtol=0,atol=0)
    torch.testing.assert_close(out[~clean],.8*z[~clean]+.78*eps[~clean])
    out.sum().backward()
    torch.testing.assert_close(z.grad[clean],torch.ones_like(z.grad[clean]))
    torch.testing.assert_close(z.grad[~clean],torch.full_like(z.grad[~clean],.8))
    torch.testing.assert_close(module.noisy_condition(z,eps),z)
    print('PASS: exact trained E/D/G mapping, fixed .8/.6*s_ref corruption and 20% clean path',flush=True)
    batch={'image':torch.randn(3,3,4,4),'latent':torch.randn(3,4,4,4),'label':torch.tensor([0,1,2])}
    fd_inputs=[]; decoder_inputs=[]
    h1=module.fd.register_forward_pre_hook(lambda m,args:fd_inputs.append(args[0]))
    h2=module.model.net.register_forward_pre_hook(lambda m,args,kw:decoder_inputs.append(kw['concat_tokens']),with_kwargs=True)
    for kind in ['recon','fd']:
        module.zero_grad(set_to_none=True)
        image_loss,token_loss,_,_,tokens,_=module.losses(batch)
        torch.testing.assert_close(fd_inputs[-1],tokens,rtol=0,atol=0)
        assert not torch.equal(decoder_inputs[-1],tokens)
        (image_loss if kind=='recon' else token_loss).backward()
        assert module.tokenizer.cls_proj.fc.weight.grad.abs().sum()>0
        assert module.tokenizer.encoder.embeddings.cls_token.grad.abs().sum()>0
        other=module.prior if kind=='recon' else module.model
        assert all(p.grad is None for p in other.parameters())
        assert all(p.grad is None for p in module.first_stage.parameters())
    h1.remove(); h2.remove()
    module.zero_grad(set_to_none=True)
    loss,metrics=module(batch)
    torch.testing.assert_close(loss,metrics['train/image_loss']+.1*metrics['train/token_loss'])
    loss.backward()
    assert all(torch.isfinite(v) and v>0 for v in module.gradient_metrics().values())
    assert all(p.grad is None for p in module.model.net.t_embedder.mlp.parameters())
    before={k:v.clone() for k,v in module.state_dict().items()}
    optimizer,scheduler=module.configure_optimizers()
    assert [g['initial_lr'] for g in optimizer.param_groups]==[1e-5,1e-6]
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in module.parameters() if p.requires_grad}
    for step in range(2):train_batch(module,batch,optimizer,scheduler,step,1,1.,torch.device('cpu'))
    for key,value in before.items():
        if key.startswith(('model.net.t_embedder.mlp.','ema_model.net.t_embedder.mlp.')):
            torch.testing.assert_close(module.state_dict()[key],value,rtol=0,atol=0)
    for prefix in ['tokenizer.','model.net.blocks.','prior.net.t_embedder.mlp.']:
        assert any(not torch.equal(value,module.state_dict()[key]) for key,value in before.items() if key.startswith(prefix))
    print('PASS: clean-only FD, separate reconstruction noise, both E gradient paths, frozen D time and optimizer coverage',flush=True)
    with tempfile.TemporaryDirectory(prefix='fd-posttrain-test-') as temporary:
        directory=Path(temporary)
        saved=directory/'saved.ckpt';save_checkpoint(saved,module,optimizer,scheduler,2,2,{})
        loaded=FDPosttrainRepTok.from_checkpoint(saved)
        assert loaded.decoder_noise_reference_std==1.3
        opt,sched=loaded.configure_optimizers();load_checkpoint(saved,loaded,opt,sched)
        module.eval();loaded.eval()
        torch.testing.assert_close(module.sample_tokens(batch['label'],3),loaded.sample_tokens(batch['label'],3),rtol=0,atol=0)
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
            cfg=compose(config_name='fd_posttrain',overrides=['trainer_module.params.decoder_noise_reference_std=1.3'])
        cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        cfg.trainer_module.target='fd_posttrain.SelectionFixture'
        cfg.trainer_module.params=mapped['hyper_parameters']
        # Start from the exact mapped pretrained miniature E/D/G, not random weights.
        initial=directory/'initial.ckpt'
        mapped['state_dict']['validation_calls']=torch.tensor(0)
        torch.save(mapped,initial)
        cfg.load_weights=str(initial)
        cfg.data={'target':'fd.FixedData','params':{'batch_size':2,'val_batch_size':2}}
        cfg.log_root=str(directory/'logs')
        cfg.trainer_params.precision='32-true';cfg.trainer_params.max_steps=2
        cfg.trainer_params.limit_val_batches=1;cfg.trainer_params.val_check_interval=2
        cfg.trainer_params.log_every_n_steps=1;cfg.checkpoint_params.every_n_train_steps=0
        OmegaConf.save(cfg,directory/'test.yaml')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'tests'),OMP_NUM_THREADS='1')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                 str(ROOT/'train.py'),'--config-path',str(directory),'--config-name','test']
        subprocess.run(command+['slurm_id=first'],cwd=ROOT,env=env,check=True)
        first=next((directory/'logs').glob('**/first/checkpoints/final.ckpt'))
        subprocess.run(command+['slurm_id=resumed','load_weights=null','trainer_params.max_steps=4',f'resume_checkpoint={first}'],cwd=ROOT,env=env,check=True)
        last=next((directory/'logs').glob('**/resumed/checkpoints/final.ckpt'))
        subprocess.run(command+['slurm_id=continuous','trainer_params.max_steps=4'],cwd=ROOT,env=env,check=True)
        continuous=next((directory/'logs').glob('**/continuous/checkpoints/final.ckpt'))
        a=torch.load(last,weights_only=False);b=torch.load(continuous,weights_only=False)
        assert a['global_step']==4 and int(a['state_dict']['fd.updates'])==4 and len(a['rng_states'])==2
        assert a['hyper_parameters']['decoder_noise_reference_std']==1.3
        for key,value in a['state_dict'].items():
            torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
        assert a['best_validation']==b['best_validation']==dict(metric='gfid',value=1.,step=2)
        assert all(int(v['step'])==4 for v in a['optimizer_states'][0]['state'].values())
        for row in csv.DictReader((last.parent.parent/'metrics.csv').open()):
            if row['train/loss']:
                assert abs(float(row['train/loss'])-float(row['train/image_loss'])-.1*float(row['train/token_loss']))<1e-5
        print('PASS: actual clean/fresh-FD validation, native 2-rank resume, exact uninterrupted parity, best-checkpoint selection',flush=True)


if __name__=='__main__':main()
