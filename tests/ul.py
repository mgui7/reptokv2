"""UL loss equivalence, gradient routing, baseline architecture and DDP resume."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import model_params
from ldm.models.transformer.dit import DiT
from ldm.ul import ULRepTok
from ldm.ul_objective import vp, flow_input, elbo_velocity_loss, terminal_kl, sample_tokens
from ldm.train_utils import save_checkpoint, seed_everything


def main():
    torch.set_num_threads(1)
    seed_everything(17)
    # Compare both value AND full-target/input gradients against Eq. 3 x-prediction.
    z = torch.randn(5, 1, 8, requires_grad=True)
    eps = torch.randn_like(z)
    lam = torch.tensor([-10., -3., 0., 2., 5.])
    noisy, u = flow_input(z, eps, lam)
    velocity = .3 * noisy.sin()
    estimate = noisy + (1-u[:, None, None]) * velocity
    direct = (.5 * 15 * lam.exp() * (estimate-z).square().flatten(1).sum(1) / 64).mean()
    converted = elbo_velocity_loss(velocity, z, eps, lam, 15., 64)
    torch.testing.assert_close(converted, direct, rtol=3e-6, atol=1e-7)
    torch.testing.assert_close(torch.autograd.grad(converted, z, retain_graph=True)[0],
                               torch.autograd.grad(direct, z)[0], rtol=1e-5, atol=1e-7)
    alpha, sigma = vp(torch.tensor(-10.))
    expected_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(alpha*z.detach(), sigma),
        torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))).flatten(1).sum(1).mean()/64
    torch.testing.assert_close(terminal_kl(z, -10., 64), expected_kl, atol=1e-8, rtol=1e-3)
    # Constant velocity oracle: exact Euler path and finite-noise VP endpoint.
    source = torch.randn(3, 1, 8)
    v = torch.randn_like(source)
    a,s = vp(torch.tensor([-10.,5.])); endpoints=a/(a+s)
    expected=(source/(a[0]+s[0])+(endpoints[1]-endpoints[0])*v)*(a[1]+s[1])
    torch.testing.assert_close(sample_tokens(source,lambda x,t:v,-10.,5.,7),expected)
    print('PASS: UL ELBO value/gradient equivalence, terminal KL and sampler endpoint', flush=True)

    with tempfile.TemporaryDirectory(prefix='ul-test-') as directory:
        directory=Path(directory)
        base=OmegaConf.to_container(model_params(),resolve=True)
        net=base['flow_cfg']['params']['net_cfg']['params']
        donor=DiT(**{k:v for k,v in net.items() if k!='null_class_cond'})
        with torch.no_grad():
            for block in donor.blocks:
                block.adaLN_modulation[-1].weight.normal_(std=.1)
                block.adaLN_modulation[-1].bias.normal_(std=.1)
            donor.final_layer.linear.weight.normal_(std=.1)
        initializer=directory/'sit.pt'; torch.save(donor.state_dict(),initializer)
        net['load_from_ckpt']=str(initializer)
        params={k:base[k] for k in ['flow_cfg','tokenizer_cfg','first_stage_cfg','lr_scheduler_cfg']}
        params.update(lr=.001, ema_rate=.9, prior_weight=.1, token_steps=3, decoder_steps=3)
        params['prior_cfg']={'target':'ldm.generation.TokenFlow','params':{
            'net_cfg':{'target':'ldm.models.transformer.token_dit.OneDimensionalTransformer','params':{
                'num_channels':1,'in_channels':32,'hidden_size':32,'depth':2,'num_heads':4,
                'num_classes':3,'class_dropout_prob':.1,'n_registers':1,'split_channels':1}}}}
        module=ULRepTok(**params)
        batch={'image':torch.randn(3,3,4,4),'latent':torch.randn(3,4,4,4),'label':torch.tensor([0,1,2])}
        groups=[{p.data_ptr() for p in m.parameters()} for m in [module.model,module.prior,module.tokenizer]]
        assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
        for kind in ['recon','prior']:
            module.zero_grad(set_to_none=True)
            image_loss,prior_loss,_,tokens=module.losses(batch)
            tokens.retain_grad()
            (image_loss if kind=='recon' else prior_loss).backward()
            assert tokens.grad is not None and tokens.grad.abs().sum()>0
            assert module.tokenizer.cls_proj.fc.weight.grad.abs().sum()>0
            assert module.tokenizer.encoder.embeddings.cls_token.grad.abs().sum()>0
            other=module.prior if kind=='recon' else module.model
            assert all(p.grad is None for p in other.parameters())
            assert all(p.grad is None for p in module.first_stage.parameters())
        module.zero_grad(set_to_none=True)
        loss,metrics=module(batch)
        torch.testing.assert_close(loss,metrics['train/image_loss']+.1*metrics['train/token_loss'])
        loss.backward()
        assert all(p.grad is not None for p in module.parameters() if p.requires_grad)
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.requires_grad)
        optimizer,scheduler=module.configure_optimizers(); optimizer.step(); scheduler.step()
        path=directory/'saved.ckpt'; save_checkpoint(path,module,optimizer,scheduler,1,1,{})
        initializer.unlink()
        module.eval(); loaded=ULRepTok.from_checkpoint(path)
        expected=module.sample_images(module.sample_tokens(batch['label'],37),41)
        actual=loaded.sample_images(loaded.sample_tokens(batch['label'],37),41)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        print('PASS: independent modules, prior/recon encoder gradients, frozen weights, portable samples',flush=True)

        cfg=OmegaConf.load(ROOT/'configs/ul.yaml')
        cfg.defaults=[{'override hydra/hydra_logging':'disabled'},{'override hydra/job_logging':'disabled'}]
        cfg.trainer_module.params=loaded.hyper_parameters
        cfg.data={'target':'fixtures.TinyGenData','params':{'batch_size':2,'val_batch_size':2}}
        cfg.log_root=str(directory/'logs')
        cfg.trainer_params.precision='32-true'
        cfg.trainer_params.max_steps=2
        cfg.trainer_params.limit_val_batches=0
        cfg.trainer_params.log_every_n_steps=1
        cfg.checkpoint_params.every_n_train_steps=0
        OmegaConf.save(cfg,directory/'test.yaml')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'tests'),OMP_NUM_THREADS='1')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                 str(ROOT/'train.py'),'--config-path',str(directory),'--config-name','test']
        subprocess.run(command+['slurm_id=first'],cwd=ROOT,env=env,check=True)
        first=next((directory/'logs').glob('**/first/checkpoints/final.ckpt'))
        subprocess.run(command+['slurm_id=resumed','trainer_params.max_steps=4',f'resume_checkpoint={first}'],
                       cwd=ROOT,env=env,check=True)
        last=next((directory/'logs').glob('**/resumed/checkpoints/final.ckpt'))
        state=torch.load(last,weights_only=False)
        assert state['global_step']==4 and len(state['rng_states'])==2
        assert state['hyper_parameters']['prior_weight']==.1
        assert state['hyper_parameters']['flow_cfg']['params']['net_cfg']['params']['load_from_ckpt'] is None
        assert all(v['step'].item()==4 for v in state['optimizer_states'][0]['state'].values())
        assert all(torch.isfinite(v).all() for v in state['state_dict'].values())
        for row in csv.DictReader((last.parent.parent/'metrics.csv').open()):
            if row.get('train/loss'):
                assert abs(float(row['train/loss'])-float(row['train/image_loss'])-.1*float(row['train/token_loss']))<1e-5
        print('PASS: two-process DDP training/resume, optimizer/RNG and objective preserved',flush=True)


if __name__=='__main__':
    main()
