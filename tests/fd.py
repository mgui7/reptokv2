"""FD mathematics, population DDP gradients, joint routing and portable resume."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from torch import distributed as dist, nn
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import model_params, TinyGenData
from ldm.models.transformer.dit import DiT
from ldm.fd import FDRepTok
from ldm.fd_objective import PopulationFD, frechet
from ldm.train_utils import save_checkpoint, seed_everything, wrap_ddp


class FixedData(TinyGenData):
    # A fixed stream isolates model/optimizer/RNG resume from the native loader's
    # intentionally uncheckpointed dataset cursor.
    def train_dataloader(self):
        batch = next(iter(super().train_dataloader()))
        def repeat():
            while True:
                yield batch
        return repeat()


def ddp_probe():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    torch.manual_seed(9)
    real, fake = torch.randn(12, 4, dtype=torch.double), torch.randn(12, 4, dtype=torch.double)
    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.eye(4, dtype=torch.double) * .7)
            self.b = nn.Parameter(torch.eye(4, dtype=torch.double) * 1.3)
            self.fd = PopulationFD(4)
        def forward(self, x, y):
            return self.fd(x @ self.a, y @ self.b)[0]
    reference = Probe()
    # Build full-population reference before wrapping: bypass distributed gather.
    original = dist.is_initialized
    dist.is_initialized = lambda: False
    reference(real, fake).backward()
    dist.is_initialized = original
    distributed = Probe()
    wrapped = wrap_ddp(distributed, torch.device('cpu'))
    wrapped(real.chunk(2)[rank], fake.chunk(2)[rank]).backward()
    for name in ('a', 'b'):
        torch.testing.assert_close(getattr(distributed, name).grad, getattr(reference, name).grad,
                                   rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(distributed.fd.means, reference.fd.means, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(distributed.fd.seconds, reference.fd.seconds, rtol=1e-12, atol=1e-12)
    dist.destroy_process_group()
    print('PASS: DDP gradient and population statistics equal concatenated global batch', flush=True)


def main():
    torch.set_num_threads(1)
    seed_everything(17)
    # Gradcheck with covariance parameterized SPD so finite differences stay symmetric.
    mean_a, mean_b = [torch.randn(4, dtype=torch.double, requires_grad=True) for _ in range(2)]
    a, b = [torch.randn(4, 4, dtype=torch.double, requires_grad=True) for _ in range(2)]
    eye = torch.eye(4, dtype=torch.double)
    objective = lambda ma, aa, mb, bb: frechet(ma, aa @ aa.T + eye, mb, bb @ bb.T + eye)
    assert torch.autograd.gradcheck(objective, (mean_a, a, mean_b, b), atol=2e-5, rtol=2e-4)
    # Repeated eigenvalues must not create NaN gradients.
    aa = (eye * 2).requires_grad_(); bb = (eye * 3).requires_grad_()
    value = frechet(mean_a.detach()*0, aa, mean_b.detach()*0, bb, jitter=1e-5)
    value.backward()
    expected = 4 * ((2+1e-5)**.5-(3+1e-5)**.5)**2
    torch.testing.assert_close(value, value.new_tensor(expected))
    assert torch.isfinite(aa.grad).all() and torch.isfinite(bb.grad).all()
    # Singular batch covariance, permutation invariance and history detached.
    real = torch.randn(3, 8, requires_grad=True); fake = torch.randn(3, 8, requires_grad=True)
    fd = PopulationFD(8)
    loss, raw, _ = fd(real, fake)
    perm = PopulationFD(8)(real.flip(0), fake.roll(1, 0))[1]
    torch.testing.assert_close(raw, perm)
    loss.backward()
    assert real.grad.abs().sum()>0 and fake.grad.abs().sum()>0
    assert not fd.means.requires_grad and not fd.seconds.requires_grad
    old_mean, old_second = fd.means.clone(), fd.seconds.clone()
    next_real, next_fake = torch.randn(3, 8, requires_grad=True), torch.randn(3, 8, requires_grad=True)
    old_grad = real.grad.clone()
    fd(next_real, next_fake)[0].backward()
    torch.testing.assert_close(real.grad, old_grad)
    features = torch.stack([next_real, next_fake]).double()
    torch.testing.assert_close(fd.means, .99*old_mean+.01*features.mean(1))
    torch.testing.assert_close(fd.seconds, .99*old_second+.01*(features.transpose(1,2)@features/3))
    print('PASS: FD finite-difference gradients, repeated/singular spectra, permutation and detached EMA', flush=True)

    with tempfile.TemporaryDirectory(prefix='fd-test-') as directory:
        directory = Path(directory)
        base = OmegaConf.to_container(model_params(), resolve=True)
        net = base['flow_cfg']['params']['net_cfg']['params']
        donor = DiT(**{k:v for k,v in net.items() if k!='null_class_cond'})
        with torch.no_grad():
            for block in donor.blocks:
                block.adaLN_modulation[-1].weight.normal_(std=.1)
                block.adaLN_modulation[-1].bias.normal_(std=.1)
            donor.final_layer.linear.weight.normal_(std=.1)
        initializer=directory/'sit.pt'; torch.save(donor.state_dict(),initializer)
        net['load_from_ckpt']=str(initializer)
        params={k:base[k] for k in ['flow_cfg','tokenizer_cfg','first_stage_cfg','lr_scheduler_cfg']}
        params.update(lr=.001, ema_rate=.9, fd_weight=.1, token_steps=1, decoder_steps=3)
        params['prior_cfg']={'target':'ldm.generation.TokenFlow','params':{
            'net_cfg':{'target':'ldm.models.transformer.token_dit.OneDimensionalTransformer','params':{
                'num_channels':1,'in_channels':32,'hidden_size':32,'depth':2,'num_heads':4,
                'num_classes':3,'class_dropout_prob':.1,'n_registers':1,'split_channels':1}}}}
        module=FDRepTok(**params)
        batch={'image':torch.randn(3,3,4,4),'latent':torch.randn(3,4,4,4),'label':torch.tensor([0,1,2])}
        calls=[]
        hook=module.prior.register_forward_pre_hook(lambda m,args: calls.append((args[0].detach().clone(),args[1].detach().clone())))
        for kind in ['recon','fd']:
            module.zero_grad(set_to_none=True)
            image_loss, token_loss, *_ = module.losses(batch)
            (image_loss if kind=='recon' else token_loss).backward()
            assert module.tokenizer.cls_proj.fc.weight.grad.abs().sum()>0
            assert module.tokenizer.encoder.embeddings.cls_token.grad.abs().sum()>0
            other=module.prior if kind=='recon' else module.model
            assert all(p.grad is None for p in other.parameters())
            assert all(p.grad is None for p in module.first_stage.parameters())
        assert len(calls)==2 and all(torch.count_nonzero(t)==0 for _,t in calls)
        module.zero_grad(set_to_none=True)
        loss, metrics=module(batch)
        torch.testing.assert_close(loss,metrics['train/image_loss']+.1*metrics['train/token_loss'])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters() if p.requires_grad)
        optimizer,scheduler=module.configure_optimizers(); optimizer.step(); scheduler.step()
        path=directory/'saved.ckpt'; save_checkpoint(path,module,optimizer,scheduler,1,1,{})
        initializer.unlink()
        module.eval(); loaded=FDRepTok.from_checkpoint(path)
        calls.clear()
        expected=module.sample_tokens(batch['label'],37)
        assert len(calls)==1 and torch.count_nonzero(calls[0][1])==0
        torch.testing.assert_close(loaded.sample_tokens(batch['label'],37),expected,rtol=0,atol=0)
        assert loaded.fd.updates.item()==module.fd.updates.item()
        torch.testing.assert_close(loaded.fd.seconds,module.fd.seconds,rtol=0,atol=0)
        hook.remove()
        print('PASS: single pure-noise call, two-sided FD/recon routing, frozen weights and portable population state',flush=True)

        cfg=OmegaConf.load(ROOT/'configs/fd.yaml')
        cfg.defaults=[{'override hydra/hydra_logging':'disabled'},{'override hydra/job_logging':'disabled'}]
        cfg.trainer_module.params=loaded.hyper_parameters
        cfg.data={'target':'fd.FixedData','params':{'batch_size':2,'val_batch_size':2}}
        cfg.log_root=str(directory/'logs')
        cfg.trainer_params.precision='32-true'
        cfg.trainer_params.max_steps=2
        cfg.trainer_params.limit_val_batches=0
        cfg.trainer_params.log_every_n_steps=1
        cfg.checkpoint_params.every_n_train_steps=0
        OmegaConf.save(cfg,directory/'test.yaml')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'tests'),OMP_NUM_THREADS='1')
        launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2']
        subprocess.run(launcher+[str(ROOT/'tests/fd.py'),'ddp_probe'],cwd=ROOT,env=env,check=True)
        command=launcher+[str(ROOT/'train.py'),'--config-path',str(directory),'--config-name','test']
        subprocess.run(command+['slurm_id=first'],cwd=ROOT,env=env,check=True)
        first=next((directory/'logs').glob('**/first/checkpoints/final.ckpt'))
        subprocess.run(command+['slurm_id=resumed','trainer_params.max_steps=4',f'resume_checkpoint={first}'],cwd=ROOT,env=env,check=True)
        last=next((directory/'logs').glob('**/resumed/checkpoints/final.ckpt'))
        state=torch.load(last,weights_only=False)
        assert state['global_step']==4 and len(state['rng_states'])==2
        assert state['hyper_parameters']['fd_weight']==.1
        assert state['state_dict']['fd.updates'].item()==4
        assert all(v['step'].item()==4 for v in state['optimizer_states'][0]['state'].values())
        assert all(torch.isfinite(v).all() for v in state['state_dict'].values())
        for row in csv.DictReader((last.parent.parent/'metrics.csv').open()):
            if row.get('train/loss'):
                assert abs(float(row['train/loss'])-float(row['train/image_loss'])-.1*float(row['train/token_loss']))<1e-5
        # Same seed, uninterrupted run must match resumed model, EMA and optimizer.
        subprocess.run(command+['slurm_id=continuous','trainer_params.max_steps=4'],cwd=ROOT,env=env,check=True)
        continuous=next((directory/'logs').glob('**/continuous/checkpoints/final.ckpt'))
        expected=torch.load(continuous,weights_only=False)
        for key,value in state['state_dict'].items():
            torch.testing.assert_close(value,expected['state_dict'][key],rtol=0,atol=0)
        print('PASS: native DDP resume, population/optimizer/RNG state, exact uninterrupted parity',flush=True)


if __name__=='__main__':
    ddp_probe() if len(sys.argv)>1 else main()
