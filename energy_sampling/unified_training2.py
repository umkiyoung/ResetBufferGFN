from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name
from energy_utils import get_energy
#from buffer import ReplayBuffer
#from RND_buffer import ReplayBuffer
from max_reward_buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from gflownet_losses import *
from evaluations import *
import time

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--buffer_size', type=int, default=300 * 1000)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='9gmm',
                    choices=('9gmm', '25gmm', 'hard_funnel', 'easy_funnel', 'many_well'))
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--both_ways', action='store_true', default=False)


# For Phase model
parser.add_argument('--only_fwd', action='store_true', default=False)
parser.add_argument('--only_bwd', action='store_true', default=False)
parser.add_argument('--phase1', type=int, default=15000)
parser.add_argument('--phase2', type=int, default=25000)
parser.add_argument('--phase3', type=int, default=25000)
parser.add_argument('--phase2reset', action='store_true', default=False)
parser.add_argument('--phase1_spreading', action='store_true', default=False)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

parser.add_argument('--ld_schedule', action='store_true', default=False)

# target acceptance rate
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


# For replay buffer
################################################################
parser.add_argument('--coreset_size', type=int, default=30000)
parser.add_argument('--load_buffer_path', type=str, default=None)
parser.add_argument('--pr_or_co', type=str, default="prioritize", choices=('prioritize', 'coreset'))
################################################################

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--load_model_path", type=str, default=None)

parser.add_argument('--wandb', action='store_true', default=False)
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 2000
final_eval_data_size = 10000
plot_data_size = 2000
final_plot_data_size = 10000

if args.pis_architectures:
    args.zero_init = True
    
if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)


def plot_step(energy, gfn_model=None, name="", buffer=None, plot_size=None, is_buffer="buffer"):
    if is_buffer == "buffer":
        # Buffer에서 샘플링
        samples, _ = buffer.sample(plot_size)
        prefix = "buffer"
    elif is_buffer == "filter":
        samples = buffer.truncate(plot_size, return_samples=True)
        prefix = "filter"        
    else:
        # GFN 모델에서 샘플링
        samples = gfn_model.sample(plot_size, energy.log_reward)
        prefix = ""

    # Visualization 및 파일 저장
    if args.energy == 'many_well':
        vizualizations = viz_many_well(energy, samples)
        fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23 = vizualizations

        fig_samples_x13.savefig(f'{name}{prefix}samplesx13.pdf', bbox_inches='tight')
        fig_samples_x23.savefig(f'{name}{prefix}samplesx23.pdf', bbox_inches='tight')

        fig_kde_x13.savefig(f'{name}{prefix}kdex13.pdf', bbox_inches='tight')
        fig_kde_x23.savefig(f'{name}{prefix}kdex23.pdf', bbox_inches='tight')

        fig_contour_x13.savefig(f'{name}{prefix}contourx13.pdf', bbox_inches='tight')
        fig_contour_x23.savefig(f'{name}{prefix}contourx23.pdf', bbox_inches='tight')

        return {
            f"visualization/{prefix}contourx13": wandb.Image(fig_to_image(fig_contour_x13)),
            f"visualization/{prefix}contourx23": wandb.Image(fig_to_image(fig_contour_x23)),
            f"visualization/{prefix}kdex13": wandb.Image(fig_to_image(fig_kde_x13)),
            f"visualization/{prefix}kdex23": wandb.Image(fig_to_image(fig_kde_x23)),
            f"visualization/{prefix}samplesx13": wandb.Image(fig_to_image(fig_samples_x13)),
            f"visualization/{prefix}samplesx23": wandb.Image(fig_to_image(fig_samples_x23))
        }

    elif energy.data_ndim != 2:
        return {}

    else:
        fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
        fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), n_contour_levels=150, device=device)
        plot_kde(samples, ax=ax_kde_overlay, bounds=(-13., 13.))
        plot_kde(samples, ax=ax_kde, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_contour, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-13., 13.))

        fig_contour.savefig(f'{name}{prefix}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}{prefix}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}{prefix}kde.pdf', bbox_inches='tight')

        return {
            f"visualization/{prefix}contour": wandb.Image(fig_to_image(fig_contour)),
            f"visualization/{prefix}kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
            f"visualization/{prefix}kde": wandb.Image(fig_to_image(fig_kde))
        }


def eval_step(eval_data, energy, gfn_model, final_eval=False):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics['final_eval/log_Z'], metrics['final_eval/log_Z_lb'], metrics[
            'final_eval/log_Z_learned'], metrics['eval/eubo'] = log_partition_function(
            init_state, gfn_model, energy.log_reward, target_energy=energy, gt_xs=eval_data)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
            'eval/log_Z_learned'], metrics['eval/eubo'] = log_partition_function(
            init_state, gfn_model, energy.log_reward, target_energy=energy, gt_xs=eval_data)
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    else:
        if final_eval:
            metrics['final_eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                              gfn_model,
                                                                                                              energy.log_reward)
        else:
            metrics['eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                        gfn_model,
                                                                                                        energy.log_reward)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
    gfn_model.train()
    return metrics


def train_step(energy, gfn_model, gfn_optimizer, it, exploratory, buffer, buffer_ls, exploration_factor, exploration_wd, phase1 =True):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0 or args.only_fwd:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r  = fwd_train_step(energy, gfn_model, exploration_std, return_exp=True)
                if phase1 == True:
                    buffer.add(states[:, -1],log_r)
                    # reset loss to 0
                    loss = 0.0
                    return
            else:
                loss = fwd_train_step(energy, gfn_model, exploration_std)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std, it=it)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std, it=it)
    else:
        loss = fwd_train_step(energy, gfn_model, exploration_std)
    
    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(energy, gfn_model, exploration_std, return_exp=False):
    init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix,
                                exploration_std=exploration_std, return_exp=return_exp)
    return loss


def bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std=None, it=0):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.batch_size, exploration_std).to(device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(device)
    elif args.sampling == 'buffer':
        if args.local_search:
            if it % args.ls_cycle < 2:
                samples, rewards = buffer.sample(args.batch_size)
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer_ls.add(local_search_samples, log_r)
        
            samples, rewards = buffer_ls.sample(args.batch_size)
        else:
            samples, rewards = buffer.sample(args.batch_size)

    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, rewards,
                                 exploration_std=exploration_std)
    return loss

def initialize_GFN(energy, args, device):
    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                        trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                        langevin=args.langevin, learned_variance=args.learned_variance,
                        partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                        pb_scale_range=args.pb_scale_range,
                        t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                        conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                        pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                        joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)
    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                    args.conditional_flow_model, args.use_weight_decay, args.weight_decay)
    return gfn_model, gfn_optimizer


def train():
    # For general training
    trial = 0
    name = f"unified_training2/{trial}/"
    while os.path.exists(name):
        trial += 1
        name = f"unified_training2/{trial}/"
    
    if not os.path.exists('name'):
        os.makedirs(name)
        os.makedirs(name + 'buffer')
    
    energy = get_energy(args.energy, device)
    eval_data = energy.sample(eval_data_size).to(device)
    final_eval_data = energy.sample(final_eval_data_size).to(device)
    
    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    
    if args.wandb:
        wandb.init(project = "GFN_v2", config=config, name=name)
        
    metrics = dict()
    buffer = ReplayBuffer(args.buffer_size, device, coreset_size=args.coreset_size, exploration_mode=True)
    buffer_ls = ReplayBuffer(args.buffer_size, device, coreset_size=args.coreset_size, exploration_mode=True)
    
    
    # ----------------- Phase 1 ----------------- #
    args.both_ways = True

    #if args.phase1_spreading:
    #    args.exploration_wd = False
    
    ## Initialize GFN
    #gfn_model, gfn_optimizer = initialize_GFN(energy, args, device)
    #gfn_model.train()
    #print("Phase 1")
    #
    #os.makedirs(f"{name}buffer/{args.energy}", exist_ok=True)
    #
    #for i in trange(args.phase1):
    #    metrics['train/loss'] = train_step(energy, gfn_model, gfn_optimizer, i, args.exploratory, buffer, buffer_ls,\
    #                                         args.exploration_factor, args.exploration_wd, phase1=True, phase1_spreading = args.phase1_spreading)
    #    if i == args.phase1 - 1:
    #        buffer.save_buffer(f"{name}buffer/{args.energy}/buffer_{args.phase1}")
    #        if args.local_search:
    #            buffer_ls.save_buffer(f"{name}buffer/{args.energy}/buffer_ls_{args.phase1}")
    #            
    #    if i % 1000 == 0 or i == args.phase1 - 1 and args.wandb:
    #        metrics.update(eval_step(eval_data, energy, gfn_model, final_eval=False))
    #        if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
    #            del metrics['eval/log_Z_learned']
    #        images = plot_step(energy=energy, gfn_model=gfn_model, name=name, buffer=None, plot_size=plot_data_size,
    #                        is_buffer="none")
    #        metrics.update(images)
    #        images_buffer = plot_step(energy, gfn_model, name, buffer, plot_data_size, is_buffer="buffer")
    #        metrics.update(images_buffer)           
    #        images_filter = plot_step(energy, gfn_model, name, buffer, plot_data_size, is_buffer="filter")
    #        metrics.update(images_filter)
    #        
    #        plt.close('all')
    #        wandb.log(metrics, step=i)
            
    # ----------------- Phase 2 ----------------- #
    # Reset GFN
    args.mode_bwd = 'tb'
    
    args.exploratory = False
    
    gfn_model, gfn_optimizer = initialize_GFN(energy, args, device)
    gfn_model.train()
    print("Phase 2")
    
    buffer.load_buffer(f"/home/uky/repos_python/Research/gfn-diffusion/unified_training/39/buffer/many_well/buffer_15000.npy")
    buffer.set_prioritization("reward")
    
    
    for i in trange(args.phase2):
        metrics['train/loss'] = train_step(energy, gfn_model, gfn_optimizer, i, args.exploratory, buffer, buffer_ls,\
                                                args.exploration_factor, args.exploration_wd, phase1=False)
        if i == args.phase2 - 1:
            # Save model
            torch.save(gfn_model.state_dict(), f'{name}model_{args.energy}_{args.phase2}_phase2.pt')
        
        if i % 1000 == 0 or i == args.phase2 - 1 and args.wandb:
            metrics.update(eval_step(eval_data, energy, gfn_model, final_eval=False))
            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']
            images = plot_step(energy=energy, gfn_model=gfn_model, name=name, buffer=None, plot_size=plot_data_size,
                            is_buffer="none")
            metrics.update(images)
            images_buffer = plot_step(energy, gfn_model, name, buffer, plot_data_size, is_buffer="buffer")
            metrics.update(images_buffer)           

            
            plt.close('all')
            wandb.log(metrics, step=i + args.phase1)
            
    # ----------------- Phase 3 ----------------- #    
    # Final evaluation
    eval_results = eval_step(final_eval_data, energy, gfn_model, final_eval=True)
    if args.wandb:
        wandb.log(eval_results)
        if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
            del eval_results['final_eval/log_Z_learned']  
    torch.save(gfn_model.state_dict(), f'{name}model_{args.energy}_{args.phase3}_phase3.pt')        
       
        
if __name__ == '__main__':
    train()
