# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
#import dnnlib
from eval.IS.bird.inception.slim import dnnlib
#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None,
                       progress=None, cache=True, gen_dataset_kwargs={}, generator_as_dataset=False):
        assert 0 <= rank < num_gpus
        self.G                        = G
        self.G_kwargs                 = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs           = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus                 = num_gpus
        self.rank                     = rank
        self.device                   = device if device is not None else torch.device('cuda', rank)
        self.progress                 = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache                    = cache
        self.gen_dataset_kwargs       = gen_dataset_kwargs
        self.generator_as_dataset     = generator_as_dataset


def load_model_from_file(model_path, device):
    """
    Load the model from a checkpoint file (e.g., .pth, .pkl).
    """
    if model_path.endswith('.pth'):
        model = torch.load(model_path, map_location=device)
    elif model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unsupported model file extension: {model_path}")
    return model

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier()  # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            if urlparse(url).path.endswith('.pkl'):
                _feature_detector_cache[key] = pickle.load(f).to(device)
            else:
                _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier()  # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1)  # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items: int):
        assert (self.num_items is None) or (cur_items <= self.num_items), f"Wrong `items` values: cur_items={cur_items}, self.num_items={self.num_items}"
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_feature_stats_for_dataset(
    opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64,
    data_loader_kwargs=None, max_items=None, temporal_detector=False, use_image_dataset=False,
    feature_stats_cls=FeatureStats, **stats_kwargs):

    dataset_kwargs = video_to_image_dataset_kwargs(opts.dataset_kwargs) if use_image_dataset else opts.dataset_kwargs
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs,
                    stats_kwargs=stats_kwargs, batch_size=batch_size, max_items=max_items)
        cache_file = os.path.join(os.path.dirname(__file__), f"{hashlib.sha256(pickle.dumps(args)).hexdigest()}.pkl")
        if os.path.exists(cache_file):
            return FeatureStats.load(cache_file)

    # Load feature detector
    device = opts.device
    print(f'Loading feature detector from {detector_url}...')
    feature_detector = get_feature_detector(detector_url, device=device, num_gpus=opts.num_gpus, rank=opts.rank)

    # Ensure that G is a model and not a string
    if isinstance(opts.G, str):
        opts.G = load_model_from_file(opts.G, device)

    print(f'Using generator: {opts.G}')
    print(f'Processing {dataset.num_samples} samples with batch size {batch_size}.')


@torch.no_grad()
def compute_feature_stats_for_generator(
    opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size: int=16,
    batch_gen=None, jit=False, temporal_detector=False, num_video_frames: int=16,
    feature_stats_cls=FeatureStats, subsample_factor: int=1, **stats_kwargs):
    
    # Set default batch_gen value if not provided
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Ensure opts.G is a model (not a string/file path)
   

    # Setup generator
    #G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    #dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Image generation function
    def run_generator(z, c, t):
        img = G(z=z, c=c, t=t, **opts.G_kwargs)
        bt, c, h, w = img.shape

        # Handle temporal images (video-like data)
        if temporal_detector:
            img = img.view(bt // num_video_frames, num_video_frames, c, h, w)  # [batch_size, t, c, h, w]
            img = img.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, c, t, h, w]

        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # Convert to uint8 image
        return img

    # JIT compilation (optional)
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        t = torch.zeros([batch_gen, G.cfg.sampling.num_frames_per_video], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c, t], check_trace=False)

    # Initialize feature statistics
    stats = feature_stats_cls(**stats_kwargs)
    assert stats.max_items is not None
    #progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)

    # Load feature detector
    #detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop for generating images and extracting features
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            cond_sample_idx = [np.random.randint(len(dataset)) for _ in range(batch_gen)]
            c = [dataset.get_label(i) for i in cond_sample_idx]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            t = [list(range(0, num_video_frames * subsample_factor, subsample_factor)) for _i in range(batch_gen)]
            t = torch.from_numpy(np.stack(t)).pin_memory().to(opts.device)
            images.append(run_generator(z, c, t))
        images = torch.cat(images)

        # Ensure RGB channels if grayscale images are generated
        if images.shape[1] == 1:
            images = images.repeat([1, 3, *([1] * (images.ndim - 2))])

        # Extract features from images
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    return stats
#-------------------------------------
    return stats

