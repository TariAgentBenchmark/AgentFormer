import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing


def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

def save_prediction(pred, data, suffix, save_dir, include_subject=False):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame = data['fut_data'], data['seq'], data['frame']
    agent_ids = data.get('agent_ids', data['valid_id'])
    subject_index = data.get('subject_index', 0)
    pred_mask = data['pred_mask']

    if include_subject:
        target_ids = agent_ids
        pred_indices = list(range(len(agent_ids)))
        mask_indices = pred_indices
    else:
        target_ids = data.get('object_ids', [agent_ids[i] for i in range(len(agent_ids)) if i != subject_index])
        pred_indices = list(range(len(target_ids)))
        mask_indices = []
        for i in range(len(target_ids)):
            orig_idx = i if i < subject_index else i + 1
            mask_indices.append(orig_idx)

    assert pred.shape[0] >= len(target_ids), "Prediction size must cover all target agents"

    for i, identity in enumerate(target_ids):    # number of agents
        pred_idx = pred_indices[i]
        orig_idx = mask_indices[i]
        if pred_mask is not None and pred_mask[orig_idx] != 1.0:
            continue
        most_recent_data = None

        """future frames"""
        traj = pred[pred_idx]
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = traj[j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

def plot_error_ellipse(data, gt_motion, sample_motion, save_path, include_subject=False, alpha_level=0.95):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Ellipse
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install via `uv pip install matplotlib`.") from exc
    if not hasattr(np, 'Inf'):
        np.Inf = np.inf

    agent_ids = data.get('agent_ids', data['valid_id'])
    subject_index = data.get('subject_index', 0)
    K = sample_motion.shape[0]
    # equal weights
    p = np.ones(K, dtype=np.float64) / K

    if include_subject:
        pairs = [(i, i, agent_ids[i]) for i in range(len(agent_ids))]
    else:
        object_ids = data.get('object_ids', [agent_ids[i] for i in range(len(agent_ids)) if i != subject_index])
        pairs = []
        for pred_idx, agent_id in enumerate(object_ids):
            if agent_id in agent_ids:
                orig_idx = agent_ids.index(agent_id)
            else:
                continue
            pairs.append((pred_idx, orig_idx, agent_id))

    if len(pairs) == 0:
        return

    # chi-square value for 2 dof
    s = 5.991 if alpha_level == 0.95 else 9.21 if alpha_level == 0.99 else 5.991

    colors = plt.cm.get_cmap('tab20', len(pairs))
    plt.figure(figsize=(6, 6))
    for idx, (pred_idx, orig_idx, agent_id) in enumerate(pairs):
        gt_final = gt_motion[orig_idx][-1].cpu().numpy()
        samples_final = sample_motion[:, pred_idx, -1, :].cpu().numpy()  # K x 2
        err = samples_final - gt_final  # K x 2

        mu = (p[:, None] * err).sum(axis=0)
        diff = err - mu
        Sigma = np.zeros((2, 2))
        for k in range(K):
            Sigma += p[k] * np.outer(diff[k], diff[k])
        Sigma += 1e-6 * np.eye(2)

        eigval, eigvec = np.linalg.eigh(Sigma)
        order = eigval.argsort()[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]
        width, height = 2 * np.sqrt(s * eigval)
        angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))

        color = colors(idx)
        plt.scatter(err[:, 0], err[:, 1], s=6, color=color, alpha=0.4)
        ell = Ellipse(xy=mu, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5,
                      label=f'ID {int(agent_id)}')
        plt.gca().add_patch(ell)

    plt.xlabel('Error X')
    plt.ylabel('Error Z')
    plt.title(f"{data['seq']} frame {int(data['frame']):06d}")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_trajectories(data, pre_motion, gt_motion, sample_motion, save_path, include_subject=False):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install via `uv pip install matplotlib`.") from exc
    if not hasattr(np, 'Inf'):
        np.Inf = np.inf

    agent_ids = data.get('agent_ids', data['valid_id'])
    subject_index = data.get('subject_index', 0)

    if include_subject:
        pairs = [(i, i, agent_ids[i]) for i in range(len(agent_ids))]
    else:
        object_ids = data.get('object_ids', [agent_ids[i] for i in range(len(agent_ids)) if i != subject_index])
        pairs = []
        for pred_idx, agent_id in enumerate(object_ids):
            if agent_id in agent_ids:
                orig_idx = agent_ids.index(agent_id)
            else:
                continue
            pairs.append((pred_idx, orig_idx, agent_id))

    if len(pairs) == 0:
        return

    colors = {
        'past': '#1f77b4',     # blue
        'pred': '#2ca02c',     # green
        'gt': '#d62728',       # red
    }
    K = sample_motion.shape[0]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    legend_flags = {'past': True, 'gt': True, 'pred': True}

    for pred_idx, orig_idx, agent_id in pairs:
        past = pre_motion[orig_idx].cpu().numpy()
        gt = gt_motion[orig_idx].cpu().numpy()
        is_subject = (include_subject and orig_idx == subject_index)

        if legend_flags['past']:
            ax.plot(past[:, 0], past[:, 1], color=colors['past'], linewidth=2, label='Past Trajectory')
            legend_flags['past'] = False
        else:
            ax.plot(past[:, 0], past[:, 1], color=colors['past'], linewidth=2)

        if legend_flags['gt']:
            ax.plot(gt[:, 0], gt[:, 1], color=colors['gt'], linewidth=2, label='GT Future Trajectory')
            legend_flags['gt'] = False
        else:
            ax.plot(gt[:, 0], gt[:, 1], color=colors['gt'], linewidth=2)

        if not is_subject:
            for k in range(K):
                pred_k = sample_motion[k, pred_idx].cpu().numpy()
                lbl = 'Predicted Future Trajectory' if legend_flags['pred'] else None
                ax.plot(pred_k[:, 0], pred_k[:, 1], color=colors['pred'], linewidth=1.5, linestyle='--', alpha=0.35, label=lbl)
                legend_flags['pred'] = False

        ax.text(past[-1, 0], past[-1, 1], f'{int(agent_id)}', fontsize=8, color='black')

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(f"{data['seq']} frame {int(data['frame']):06d}")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def test_model(generator, save_dir, cfg, include_subject=False, plot_results=False, plot_limit=10, plot_traj=False, traj_sample_idx=0):
    total_num_pred = 0
    plot_count = 0
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()

        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

        if include_subject:
            subject_idx = data.get('subject_index', 0)
            subject_traj = gt_motion_3D[subject_idx].unsqueeze(0)
            recon_motion_eval = torch.cat([subject_traj, recon_motion_3D], dim=0)
            subj_sample = subject_traj.unsqueeze(0).repeat(sample_motion_3D.shape[0], 1, 1, 1)
            sample_motion_eval = torch.cat([subj_sample, sample_motion_3D], dim=1)
        else:
            recon_motion_eval = recon_motion_3D
            sample_motion_eval = sample_motion_3D

        """save samples"""
        recon_dir = os.path.join(save_dir, 'recon'); mkdir_if_missing(recon_dir)
        sample_dir = os.path.join(save_dir, 'samples'); mkdir_if_missing(sample_dir)
        gt_dir = os.path.join(save_dir, 'gt'); mkdir_if_missing(gt_dir)
        for i in range(sample_motion_eval.shape[0]):
            save_prediction(sample_motion_eval[i], data, f'/sample_{i:03d}', sample_dir, include_subject=include_subject)
        save_prediction(recon_motion_eval, data, '', recon_dir, include_subject=include_subject)        # save recon
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir, include_subject=include_subject)              # save gt
        total_num_pred += num_pred

        if (plot_results or plot_traj) and (plot_limit <= 0 or plot_count < plot_limit):
            if plot_results:
                plot_path = os.path.join(save_dir, 'figures', f'{seq_name}_frame_{frame:06d}.png')
                plot_error_ellipse(data, gt_motion_3D, sample_motion_eval, plot_path, include_subject=include_subject)
            if plot_traj:
                pre_motion_plot = torch.stack(data['pre_motion_3D'], dim=0) * cfg.traj_scale
                plot_path_traj = os.path.join(save_dir, 'figures_traj', f'{seq_name}_frame_{frame:06d}.png')
                plot_trajectories(data, pre_motion_plot, gt_motion_3D, sample_motion_eval, plot_path_traj, include_subject=include_subject)
            plot_count += 1

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {
            'train': 32186,
            'val': 8560,
            'test': 9041
        }
        assert total_num_pred == scene_num[generator.split]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--include_subject_eval', action='store_true', default=False, help='Include subject trajectory when saving predictions for evaluation')
    parser.add_argument('--plot_results', action='store_true', default=False, help='Generate trajectory plots during evaluation')
    parser.add_argument('--plot_limit', type=int, default=10, help='Max frames to plot when --plot_results is enabled; <=0 means no limit')
    parser.add_argument('--plot_traj', action='store_true', default=False, help='Plot past / GT future / predicted future trajectories (all K samples)')
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                test_model(
                    generator,
                    save_dir,
                    cfg,
                    include_subject=args.include_subject_eval,
                    plot_results=args.plot_results,
                    plot_limit=args.plot_limit,
                    plot_traj=args.plot_traj,
                )

            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)
