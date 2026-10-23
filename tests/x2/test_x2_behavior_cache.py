"""CPU-only archive fixtures: never a surrogate production brain or MPS run."""
import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from malecns_lab.learning import x2_behavior_cache as cache
from malecns_lab.learning import x2_behavior_training as training
from malecns_lab.learning.x2_behavior_policy import X2BehaviorPolicy, BehaviorOutput
from malecns_lab.control.x2_behavior import Observation, Decision, Task, Behavior, Phase
from malecns_lab.senses.x2_behavior import channel_values


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def hashed(value, key):
    return {**value, key: training.frozen.canonical_hash(value)}


def npz(values):
    stream = io.BytesIO(); np.savez_compressed(stream, **values); return stream.getvalue()


def put(root, relative, value):
    target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value if isinstance(value, bytes) else encoded(value))


def archive_fixture(tmp_path, monkeypatch):
    """Full 50/50,818/918 inventory, with explicitly tiny state-array fixtures.

    Only test-module monkeypatching changes the audited state width; production
    has no dimensions/identity override or synthetic graph path.
    """
    root = tmp_path.resolve(); old = root/cache.CACHE_RUN; initial = root/cache.INITIAL
    monkeypatch.setattr(cache, 'NODE_COUNT', 3)
    config = training.contract('readout', seed=2401, max_updates=200, train_episodes=50,
        validation_episodes=50, wall_seconds=600, spike_backward_gain=.1)
    mapping = hashed({'channel_indices': [0]*3377, 'fixture': 'not a biological mapping'}, 'input_map_id')
    policy = hashed({'input_map': mapping, 'features': {'width': 2628}, 'frame_steps': 500,
        'precondition_steps': 600, 'fixture': 'archive serialization only; no model'}, 'policy_id')
    monkeypatch.setattr(cache,'POLICY_ID',policy['policy_id']); monkeypatch.setattr(cache,'INPUT_ID',mapping['input_map_id'])
    params = {'parameter.'+name: np.zeros(shape,np.float32) for name,shape in cache.PARAMETER_SHAPES.items()}
    pbytes = npz(params)
    meta = hashed({'schema': training.CHECKPOINT_SCHEMA, 'policy': policy,
        'context': {'optimizer_steps':0,'role':'initialization'},
        'arrays': {k:cache._array_identity(v) for k,v in params.items()}, 'arrays_sha256':cache._sha(pbytes)},'checkpoint_id')
    monkeypatch.setattr(cache,'INITIAL_ID',meta['checkpoint_id'])
    put(initial,'checkpoint.json',meta); put(initial,'arrays.npz',pbytes)
    checkpoint = {'path':str(initial),'checkpoint_id':meta['checkpoint_id'],
        'metadata_sha256':cache._sha(encoded(meta)),'arrays_sha256':cache._sha(pbytes)}
    put(old,'policy.json',policy);put(old,'input_map.json',mapping)
    # Exact current numerical/readout AST; changed orchestration comment only.
    source = Path(training.__file__).read_bytes()
    originals = {'learning/x2_behavior_training.py':source,'experiments/x2_behavior_training.py':b'# old CLI\n',
        'learning/x2_behavior_policy.py':b'# unchanged policy fixture\n'}
    current = {**originals, 'learning/x2_behavior_training.py':source+b'\n# cache orchestration fixture\n',
        'learning/x2_behavior_cache.py':b'# explicitly added cache fixture\n'}
    for name,content in originals.items(): put(old,'sources/malecns_lab/'+name,content)
    for name,content in current.items(): put(root,'src/malecns_lab/'+name,content)
    sources = {k:cache._sha(v) for k,v in current.items()}
    states = {'v':np.zeros(3,np.float32),'g':np.zeros(3,np.float32),'refractory_until':np.zeros(3,np.int64),
        'phase':np.zeros(3377,np.float32),'event_counts':np.zeros(3377,np.int64),
        **{f'pending.{i:02d}':np.zeros(3,np.float32) for i in range(18)}}
    sbytes=npz(states); declarations={k:cache._array_identity(v) for k,v in states.items()}
    splits={};selected={'locked_test_opened':False}; at=100.
    for split in ('train','validation'):
        episodes=[]
        for i in range(50):
            task=list(Task)[i%5]; eid=hashlib.sha256(f'{split}{i}'.encode()).hexdigest()
            count=9 if i < (16 if split=='train' else 2) else 8
            rows=[]
            for index in range(count):
                observation=Observation(eid,eid,index,eid+'-request',task,float(index),request_changed=index==0)
                rows.append({'frame_index':index,'observation':observation.to_dict(),
                    'target':{'behavior':'HOLD','phase':'START','intensity':.5}})
            episode={'episode_id':eid,'family_id':f'{split}-{i}','split':split,'task':str(task),
                'frames':rows,'sequence_sha256':training.frozen.canonical_hash(rows)}
            episodes.append(episode)
            state_ids=[];relative=split+'-features/'+eid
            for index in range(count+1):
                smeta=hashed({'schema':'malecns_lab.x2_behavior_state.v1','policy_id':policy['policy_id'],
                    'session_id':eid,'frame_index':index,'observation_seq':index-1,'neural_step':600+index*500,
                    'arrays':declarations},'state_id')
                state_ids.append(smeta['state_id'])
                put(old,relative+f'/state-{index:04d}.json',{**smeta,'file_sha256':cache._sha(sbytes)})
                put(old,relative+f'/state-{index:04d}.npz',sbytes)
            for index,row in enumerate(rows):
                observation=Observation.from_dict(row['observation'])
                values=tuple(channel_values(observation).values())
                arr={'encoded_input':np.asarray([values[0]]*3377,np.float32),'features_mv':np.full(2628,-50.,np.float32),
                    'behavior_logits':np.zeros(5,np.float32),'phase_logits':np.zeros(3,np.float32),'parameter_raw':np.zeros(1,np.float32)}
                fbytes=npz(arr)
                decision=Decision(eid,eid,index,observation.request_id,index,state_ids[index],state_ids[index+1],
                    tuple(arr['behavior_logits'].tolist()),tuple(arr['phase_logits'].tolist()),(0.,),Behavior.HOLD,Phase.START,.5,
                    policy['policy_id'],meta['checkpoint_id'],mapping['input_map_id'],600+index*500,1100+index*500)
                put(old,relative+f'/frame-{index:04d}.npz',fbytes)
                at+=1
                put(old,relative+f'/frame-{index:04d}.json',{'arrays_sha256':cache._sha(fbytes),'observation':row['observation'],
                    'decision':decision.to_dict(),'host_monotonic':at})
        splits[split]=episodes;selected[split]=[e['episode_id'] for e in episodes]
    identity=hashed({'seed':2401,'split_sha256':{k:training.frozen.canonical_hash(v) for k,v in splits.items()}},'dataset_id')
    dataset={'schema':'malecns_lab.x2_behavior_dataset.v1','identity':identity,'files':{}}
    for split,episodes in splits.items():
        body=encoded(episodes);put(old,'dataset/'+split+'.json',body)
        dataset['files'][split]={'path':split+'.json','sha256':cache._sha(body)}
    put(old,'dataset/manifest.json',dataset);put(old,'selected_episodes.json',selected)
    runtime={'fixture':'CPU archive admission; not numeric runtime'}
    reg=hashed({'output':str(old),'root':str(root),'checkpoint':checkpoint,'configuration':{**config,'wall_seconds':5400},
        'runtime':runtime,'dataset':dataset,'source_sha256':{k:cache._sha(v) for k,v in originals.items()}},'registration_id')
    monkeypatch.setattr(cache,'CACHE_REGISTRATION',reg['registration_id']);put(old,'registration.json',reg)
    put(old,'summary.json',{'status':'failed','supervisor':{'reason':'wall_limit'}})
    return SimpleNamespace(root=root,old=old,config=config,checkpoint=checkpoint,sources=sources,runtime=runtime,selected=selected)


def test_full_cache_admission_worker_recheck_and_corruption_rejection(tmp_path,monkeypatch):
    f=archive_fixture(tmp_path,monkeypatch)
    reads=[]; original_read=cache._read
    def watched(base,relative,**kwargs):
        reads.append(str(relative));assert 'locked' not in str(relative)
        return original_read(base,relative,**kwargs)
    monkeypatch.setattr(cache,'_read',watched)
    def audit(**kw):return cache.audit_cache(f.root,f.old,configuration=f.config,checkpoint=f.checkpoint,
        sources=f.sources,runtime=f.runtime,**kw)
    manifest,_,dataset=audit()
    assert manifest['frame_counts']=={'train':416,'validation':402} and manifest['state_count']==918
    assert set(dataset)=={'manifest.json','train.json','validation.json'}
    again,features,_=audit(expected=manifest,keep_features=True)
    assert again==manifest and sum(len(rows) for split in features.values() for rows in split.values())==818
    assert all(row.shape==(2628,) for split in features.values() for rows in split.values() for row in rows)
    assert not any('locked' in name for name in reads)
    first=f.old/'train-features'/f.selected['train'][0]
    def corrupt(name,edit,match):
        target=first/name; original=target.read_bytes()
        try:
            target.write_bytes(edit(original))
            with pytest.raises(ValueError,match=match):audit()
        finally:target.write_bytes(original)
    def edit_json(data,**values):return encoded({**json.loads(data),**values})
    corrupt('frame-0000.npz',lambda b:b+b'changed','NPZ hash')
    corrupt('frame-0000.json',lambda b:edit_json(b,observation={**json.loads(b)['observation'],'request_changed':False}),'observation differs')
    corrupt('frame-0000.json',lambda b:edit_json(b,decision={**json.loads(b)['decision'],'checkpoint_id':'other'}),'chain differs')
    corrupt('state-0001.json',lambda b:edit_json(b,neural_step=1200),'state identity/sequence')
    corrupt('state-0001.npz',lambda b:b+b'changed','state NPZ hash')
    # A whitespace-only metadata replacement still fails the worker's sealed
    # inventory even though its parsed values are unchanged.
    target=first/'frame-0000.json';original=target.read_bytes();target.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='after registration'):audit(expected=manifest)
    target.write_bytes(original)
    target=first/'state-0001.npz';original=target.read_bytes();target.unlink()
    with pytest.raises(ValueError,match='Missing/extra'):audit()
    target.write_bytes(original)
    extra=first/'frame-9999.json';extra.write_text('{}')
    with pytest.raises(ValueError,match='Missing/extra'):audit()
    extra.unlink()
    selection=f.old/'selected_episodes.json';original=selection.read_bytes(); changed=json.loads(original)
    changed['train'][0],changed['train'][1]=changed['train'][1],changed['train'][0];selection.write_bytes(encoded(changed))
    with pytest.raises(ValueError,match='selection/order'):audit()
    selection.write_bytes(original)
    source=f.root/'src/malecns_lab/learning/x2_behavior_policy.py';source.write_text('# numeric source changed')
    changed_sources={**f.sources,'learning/x2_behavior_policy.py':cache._sha(source.read_bytes())}
    with pytest.raises(ValueError,match='Numerical/encoder/data'):cache.audit_cache(f.root,f.old,configuration=f.config,
        checkpoint=f.checkpoint,sources=changed_sources,runtime=f.runtime)
    with pytest.raises(ValueError,match='readout-only'):cache.validate_request({**f.config,'mode':'joint'},f.checkpoint['path'],f.old,f.root)
    with pytest.raises(ValueError,match='200 updates'):cache.validate_request({**f.config,'max_updates':199},f.checkpoint['path'],f.old,f.root)
    with pytest.raises(ValueError,match='locked-test'):original_read(f.old,'dataset/locked-test.json')


@pytest.mark.parametrize('fault',['extra','nonfinite','dtype','object','shape','duplicate'])
def test_npz_inventory_and_numeric_rejection(fault):
    values={'features_mv':np.zeros(3,np.float32)};specs={'features_mv':((3,),'<f4')}
    if fault=='extra':values['teacher']=np.zeros(3,np.float32)
    if fault=='nonfinite':values['features_mv'][0]=np.nan
    if fault=='dtype':values['features_mv']=values['features_mv'].astype(np.float64)
    if fault=='object':values['features_mv']=np.asarray([{}, {}, {}],object)
    if fault=='shape':values['features_mv']=np.zeros(2,np.float32)
    content=npz(values)
    if fault=='duplicate':
        import zipfile
        stream=io.BytesIO(content)
        with zipfile.ZipFile(stream,'a') as zipped:
            with pytest.warns(UserWarning):zipped.writestr('features_mv.npy',b'bad')
        content=stream.getvalue()
    with pytest.raises((ValueError,KeyError)):cache._arrays(content,specs)


class HeadFixture(nn.Module):
    """Actual existing head function on CPU; no graph or biological simulation."""
    read_features=X2BehaviorPolicy.read_features
    def __init__(self):
        super().__init__();self.brain=nn.Module();self.brain.theta=nn.Parameter(torch.zeros(2),requires_grad=False)
        self.brain.config=SimpleNamespace(v_rest_mv=-52.,v_threshold_mv=-45.)
        self.behavior_head,self.phase_head,self.parameter_head=nn.Linear(2628,5),nn.Linear(2628,3),nn.Linear(2628,1)
        self.identity=hashed({'fixture':'CPU heads only; no model'},'policy_id')
    def validate_parameters(self):assert all(torch.isfinite(p).all() for p in self.parameters())
    def forward(self,*args):raise AssertionError('never a neural forward')
    def initial_state(self,*args):raise AssertionError('never a neural state')
    def _step(self,*args):raise AssertionError('never neural integration')


def test_worker_uses_revalidated_dataset_bytes_despite_self_consistent_run_replacement(tmp_path, monkeypatch):
    """Exercise run_worker through both real partition/selection calls, then stop.

    Only the expensive admission/graph/resource boundaries are fixtures. No
    neural computation, feature generation or optimizer update is performed.
    """
    from malecns_lab.learning import mps_memory, x2_behavior_data as data
    from malecns_lab.learning.x2_behavior_policy import PARAMETERS

    root = tmp_path.resolve()
    run = root / 'data/recovery'; run.mkdir(parents=True)
    cfg = training.contract('readout', max_updates=200, train_episodes=50,
        validation_episodes=50, wall_seconds=600, spike_backward_gain=.1)
    splits = {}
    for split in ('train', 'validation'):
        episodes = []
        # Task-grouped source order must still become the original round-robin.
        for index in range(50):
            task = list(Task)[index // 10]
            eid = hashlib.sha256(f'{split}-{index}'.encode()).hexdigest()
            frames = [{'frame_index': i, 'observation': Observation(eid, eid, i, eid+'-request',
                task, float(i), request_changed=i == 0).to_dict(),
                'target': {'behavior': 'HOLD', 'phase': 'START', 'intensity': .5}}
                for i in range(2 + index % 2)]
            episodes.append({'episode_id': eid, 'family_id': f'{split}-{index}', 'split': split,
                'task': str(task), 'frames': frames, 'sequence_sha256': training.frozen.canonical_hash(frames)})
        splits[split] = episodes

    def dataset_bytes(episodes):
        files = {name+'.json': encoded(rows) for name, rows in episodes.items()}
        manifest = {'schema': data.SCHEMA,
            'identity': hashed({'seed': 2401, 'split_sha256': {
                name: training.frozen.canonical_hash(rows) for name, rows in episodes.items()}}, 'dataset_id'),
            'files': {name: {'path': name+'.json', 'sha256': cache._sha(files[name+'.json'])} for name in episodes}}
        return {**files, 'manifest.json': encoded(manifest)}

    audited_bytes = dataset_bytes(splits)
    replaced = json.loads(encoded(splits))
    for episodes in replaced.values():
        frames = episodes[0]['frames']
        frames[0]['target']['behavior'] = 'ADVANCE'
        frames[0]['observation']['request_changed'] = False
        frames.pop()
        episodes[0]['sequence_sha256'] = training.frozen.canonical_hash(frames)
    # File hashes, dataset ID and episode sequence hashes all agree internally;
    # only the previously registered bytes identify the intended dataset.
    for name, content in dataset_bytes(replaced).items():
        put(run, 'dataset/'+name, content)
    select = training.select_episodes
    expected = {name: select(json.loads(audited_bytes[name+'.json']), 50) for name in splits}
    for name in splits:
        changed = select(data.load_split(run/'dataset', name), 50)
        assert [row['episode_id'] for row in changed] == [row['episode_id'] for row in expected[name]]
        assert changed != expected[name]

    policy = HeadFixture()
    policy.policy_id = policy.identity['policy_id']
    policy.encoder = SimpleNamespace(identity={'input_map_id': 'fixture-input'}, input_map_id='fixture-input')
    initial = {'checkpoint_id': 'fixture-initial', 'arrays': {'parameter.'+name: {} for name in PARAMETERS}}
    checkpoint_dir = root/'data/initial'; put(checkpoint_dir, 'checkpoint.json', initial)
    checkpoint = {'path': str(checkpoint_dir), 'checkpoint_id': initial['checkpoint_id'],
        'metadata_sha256': training.frozen.sha256(checkpoint_dir/'checkpoint.json')}
    manifest = hashed({'source_registration_id': 'fixture-source', 'policy_id': policy.policy_id,
        'input_map_id': policy.encoder.input_map_id, 'initial_checkpoint': checkpoint,
        'dataset_id': json.loads(audited_bytes['manifest.json'])['identity']['dataset_id']}, 'cache_id')
    put(run, 'cache_manifest.json', manifest)
    put(root, str(training.frozen.REFERENCE), b'not a graph; construction is replaced at the boundary')
    reg = hashed({'root': str(root), 'output': str(run), 'parent_pid': training.os.getppid(),
        'configuration': cfg, 'source_sha256': {}, 'runtime': {}, 'checkpoint': checkpoint,
        'dataset': json.loads(audited_bytes['manifest.json']), 'resources': {'effective_rss_limit_bytes': 1},
        'graph_sha256': training.frozen.sha256(root/training.frozen.REFERENCE),
        'feature_cache': {'cache_root': str(root/'data/old-cache'), 'cache_id': manifest['cache_id'],
            'source_registration_id': manifest['source_registration_id'],
            'initial_checkpoint_id': initial['checkpoint_id'],
            'manifest_sha256': training.frozen.sha256(run/'cache_manifest.json')}}, 'registration_id')
    put(run, 'registration.json', reg)

    audited = []
    def audit(*args, **kwargs):
        assert kwargs['expected'] == manifest and kwargs['keep_features'] is True
        audited.append(True)
        return manifest, {}, audited_bytes
    def parent_owns_lock(*args):
        raise BlockingIOError('fixture supervising parent owns lock')
    def no_disk_partition(*args):
        raise AssertionError('worker reread the untrusted new-run dataset')
    monkeypatch.setattr(cache, 'audit_cache', audit)
    monkeypatch.setattr(training.fcntl, 'flock', parent_owns_lock)
    monkeypatch.setattr(torch, 'set_num_threads', lambda *args: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda *args: None)
    monkeypatch.setattr(mps_memory, 'MPSMemorySampler', lambda *args: SimpleNamespace(
        sample=lambda *args: None, close=lambda: {'error': None}))
    monkeypatch.setattr(training.frozen, 'read_anchors', lambda *args: {'run': {'policy': {}}, 'checkpoint': {}})
    monkeypatch.setattr(training.frozen, 'load_parameters', lambda *args: {'theta': np.zeros(2, np.float32)})
    monkeypatch.setattr(X2BehaviorPolicy, 'from_reference', lambda *args, **kwargs: policy)
    monkeypatch.setattr(training, 'restore_checkpoint', lambda *args: initial)
    monkeypatch.setattr(training, 'hashlib_counts', lambda *args: 'fixture-graph-counts')
    monkeypatch.setattr(data, 'load_split', no_disk_partition)
    seen = []
    def consume_partition(rows, count):
        selected = select(rows, count)
        seen.append(selected)
        if len(seen) == 2:
            raise RuntimeError('fixture reached both worker partitions; stop before updates')
        return selected
    monkeypatch.setattr(training, 'select_episodes', consume_partition)

    result = training.run_worker(run)
    assert result['reason'] == 'RuntimeError:fixture reached both worker partitions; stop before updates'
    assert audited == [True] and result['cache_revalidated'] is True
    assert result['opened_splits'] == ['train', 'validation'] and result['optimizer_steps'] == 0
    assert seen == [expected['train'], expected['validation']]
    assert not any(result['neural_calls'].values())


def test_exact_existing_200_update_loop_fresh_parameters_earliest_selection_and_no_neural_calls(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(2401)
    policy=HeadFixture();result={'optimizer_steps':0}
    initial={name:p.detach().clone() for name,p in policy.named_parameters()}
    saved=training.save_checkpoint(tmp_path/'initial',policy,{'optimizer_steps':0,'role':'initialization'})
    with torch.no_grad():
        for p in policy.parameters():p.add_(10)
    training.restore_checkpoint(tmp_path/'initial',policy)
    assert all(torch.equal(p,initial[name]) for name,p in policy.named_parameters())
    optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    assert not optimizer.state
    restore=cache.forbid_neural_calls(HeadFixture,type(policy.brain),result)
    seen=[]; validations=[]
    train=[{'episode_id':str(i)} for i in range(50)];validation=[{'episode_id':'v'+str(i)} for i in range(50)]
    features={'sentinel':'train features'};validation_features={'sentinel':'validation features'}
    vector=torch.full((2628,),-50.)
    def cached_outputs(episodes,which):
        if which is features:
            seen.append(episodes[0]['episode_id']);heads=policy.read_features(vector)
            return {},training.loss_for(BehaviorOutput(*heads,vector,None),{'behavior':'HOLD','phase':'START','intensity':.5})
        assert episodes is validation and which is validation_features;validations.append(len(episodes))
        return {},torch.tensor(5. if result['optimizer_steps']<=100 else 3.)
    def gradient_update(loss):
        return training._audited_gradient_update(policy,optimizer,loss,directory=tmp_path/'audits'/f"{result['optimizer_steps']+1:04d}",result=result,max_updates=200)
    environment=dict(cfg={'max_updates':200},phase=lambda *a,**kw:None,train=train,optimizer=optimizer,
        cached_outputs=cached_outputs,features=features,validation=validation,validation_features=validation_features,
        gradient_update=gradient_update,torch=torch,history=[],best=float('inf'),selected=None,save_checkpoint=training.save_checkpoint,
        directory=tmp_path,policy=policy,reg={'registration_id':'cpu-loop-fixture'},write_json=training.frozen.write_json if hasattr(training.frozen,'write_json') else lambda p,v:p.write_text(json.dumps(v)))
    try:
        node=cache.readout_update_ast(Path(training.__file__).read_bytes())
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-readout-loop>','exec'),environment)
        assert result['optimizer_steps']==200 and len(environment['history'])==200
        assert seen==[str(i) for i in range(50)]*4 and validations==[50]*200
        assert environment['selected']['context']['optimizer_steps']==101
        assert torch.equal(policy.brain.theta,initial['brain.theta'])
        assert not any(result['neural_calls'].values())
        for name in ('forward','initial_state','_step'):
            with pytest.raises(ValueError,match='forbids neural call'):getattr(policy,name)()
    finally:restore()
