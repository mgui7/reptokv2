"""Check baseline config and retained computations against the original commit."""
import ast
import getpass
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
REF = '33b4277437db35929c5468feecff76b2cf224ac2'
# Apptainer --cleanenv omits USER, which only the old logging config reads.
os.environ.setdefault('USER', getpass.getuser())


def nodes(source, cls=None):
    tree = ast.parse(source)
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls) if cls else tree.body
    return {n.name: ast.dump(n, include_attributes=False) for n in body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


with tempfile.TemporaryDirectory(prefix='reptok-contract-') as temp:
    temp = Path(temp)
    archive = temp / 'old.tar'
    with archive.open('wb') as f:
        subprocess.run(['git', 'archive', REF], cwd=ROOT, stdout=f, check=True)
    old = temp / 'old'
    old.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(old, filter='data')
    with initialize_config_dir(config_dir=str(old / 'configs'), version_base=None):
        original = compose(config_name='config', overrides=['experiment=imnet256/tokenizer/cls-init-cos'])
        original = OmegaConf.to_container(original, resolve=True)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        current = OmegaConf.to_container(compose(config_name='config'), resolve=True)
    for key in ['model', 'data', 'autoencoder', 'tokenizer', 'lr_scheduler']:
        assert original[key] == current[key], key
    for key, value in current['trainer_module']['params'].items():
        assert original['trainer_module']['params'][key] == value, key
    for key, value in current['trainer_params'].items():
        assert original['trainer_params'][key] == value, key
    assert original['checkpoint_params']['every_n_train_steps'] == current['checkpoint_params']['every_n_train_steps']
    for file, cls, selected in [
        ('ldm/encoders.py', 'DINOImageEncoder', ['forward', 'encode', 'to']),
        ('ldm/tokenizer.py', None, ['Projection', 'ClsTokenizer']),
        ('ldm/dataloader.py', 'WebDataModuleFromConfig', None),
        ('ldm/flow.py', 'FlowModel', ['generate', 'compute_xt', 'compute_ut', 'training_losses', 'validation_losses']),
        ('ldm/models/transformer/dit.py', None, ['DiT', 'TimestepEmbedder', 'LabelEmbedder', 'DiTBlock', 'FinalLayer']),
    ]:
        a, b = nodes((old / file).read_text(), cls), nodes((ROOT / file).read_text(), cls)
        for name in selected or a:
            assert a[name] == b[name], (file, name)
    for file in ['ldm/metrics.py', 'ldm/models/transformer/mmdit.py']:
        assert (old / file).read_bytes() == (ROOT / file).read_bytes(), file
    print('SOURCE CONTRACT PASSED: resolved baseline settings and retained model/data/metric computations match.')
