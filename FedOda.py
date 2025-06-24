import torch
import torch.nn.functional as F
from copy import deepcopy
import math
import logging
from torch.utils.data import DataLoader
import torch.optim as optim    
import random
import itertools
from typing import List, Optional
import math
from utilities import create_balanced_loader
import numpy as np
from PPO import PPOAgent
from utils_attack import *
from attack_bo import bo_attack
from attack_bi import bi_attack
from guided_att_from_action import guided_attack_from_action
from moq_agent import MOQAgent
from moq_attack import moq_attack
from paca_agent import PCACAgent
from pcac_attack import pcac_attack

class FedOda:
    def __init__(self, model_fn, device, pooled_loader, vali_loader, total_clients_num, 
                 malicious_client_ids, fl_mali_data_dict, E=5, scaling_l2=8,
                 reduce_factor=0.8, increase_factor=1.1, threshold=0.7, 
                 client_learning_rate=0.01, client_data_size=500, client_batch_size=64,
                 logger=None):
        self.model_fn = model_fn
        self.device = device
        self.E = E # Number of rounds to consider for history
        self.client_learning_rate = client_learning_rate
        self.client_data_size = client_data_size
        self.client_batch_size = client_batch_size
        self.reduce_factor = reduce_factor
        self.increase_factor = increase_factor
        self.threshold = threshold
        self.pooled_loader = pooled_loader
        self.pooled_balanced_loader = create_balanced_loader(
                dataset=self.pooled_loader.dataset,
                batch_size=64,
                max_samples_per_class=None  # or None to auto-balance
            )
        
        self.vali_loader = vali_loader
        self.total_clients_num = total_clients_num
        self.malicious_client_ids = malicious_client_ids
        self.malicious_clients_num = len(malicious_client_ids)
        self.fl_mali_data_dict = fl_mali_data_dict
        self.current_s_m = None
        self.inference_mode = 'validation'   # 'cos' or 'l2', 'sign', 'validation', 'fixed'
        self.pooled_val = [] 
        self.val_acc_pooled_history = []
        self.attack_mode = "PCAC" # 'BO', "BI", "PPO", "ES" "MOQ", "PCAC"
        if self.attack_mode == "PPO":
            self.ppo_agent = PPOAgent(state_dim=5, action_dim=2, device=device, reward_weights=[0.0, 1.0, 0])  # 5 state features, 2 action dims
            self.ppo_episode_memory = []
        self.prev_server_model = None
        self.prev_global_update = None
        self.prev_malicious_update = None
        
        # Create the Bayesian Optimizer
        if self.attack_mode == "BO":
            l2_bound = (-1.0, 1.0)
            cos_bound = (-1.0, 1.0)
            self.bayes_optimizer = BayesOptimizer(
                bounds=[l2_bound, cos_bound],
                base_estimator="GP",
                acq_func="LCB",
                acq_func_kwargs={"kappa": 5.0},
                random_state=42
            )

        if self.attack_mode == "MOQ":
            self.moq_preference_vector = torch.tensor([0.6, 0.4], dtype=torch.float32)  # Adjust as needed
            initial_epsilon = 0.5
            self.moq_agent = MOQAgent(
                state_dim=3,          # curr_acc, acc_delta, round_norm, model_norm, avg_cos_distance
                action_dim=2,         # norm_scaling_l2, norm_scaling_cos
                reward_dim=2,
                epsilon=initial_epsilon,  # Exploration factor
                epsilon_min=0.1,
                epsilon_decay=0.9,
                epsilon_grow=1.05,
                device=self.device,
                action_grid_size=9,  # 9x9 grid for actions
                logger=logger
            )
            self.prev_state = None
            self.prev_action = None
            self.prev_acc = None
        
        if self.attack_mode == "PCAC":
            self.pcac_preference_vector = torch.tensor([0.3, 0.7], dtype=torch.float32)
            self.pcac_agent = PCACAgent(
                state_dim=3,     # same as MOQ
                action_dim=2,    # norm_scaling_l2, norm_scaling_cos
                reward_dim=2,    # r1, r2
                device=self.device,
                logger=logger
            )
            self.prev_state = None
            self.prev_action = None
            self.prev_acc = None

    
        # For tracking performance history and reward scores
        self.bo_reward_history = []  # stores rewards (with None if skipped)
        self.bo_param_history = []   # stores chosen params per round
        
        # State variables
        self.model0_sd = None
        self.modelt_sd = None
        self.att_direction = None
        self.last_update_norm = 0
        self.benign_direction = None
        self.scaling_l2 = scaling_l2
        self.epsilon = None
        self.server_model = None
        self.previous_server_model = None
        self.ema_posterior = None  # Initialize EMA tracker
        self.ema_alpha = 0.3       # Tunable: higher = more weight on recent posterior
                        
        # Logging
        self.logger = logger or logging.getLogger("FedOda")

        # Historical memory
        self.history_server_updates = []
        self.history_pred_benign_updates = []
        self.history_pred_server_updates = []
        self.history_malicious_updates = []
        self.history_malicious_selection = []
        self.posterior_history = []
        self.history_s_m = []


        # Full history for plotting/analysis
        self.full_history = {
            "s_b": [],
            "s_hat": [],
            "benign_norm": [],
            "server_update_norm": [],
            "cos_d1": [],
            "cos_d2": [],
            "cos_d3": [],
            "cos_d4": [],
            "l2_d1": [],
            "l2_d2": [],
            "l2_d3": [],
            "l2_d4": [],
            "mean_log_likelihood": [],
            "posterior": [],
            "scaling_factor": [],
            "epsilon": [],
            "norm_scaling_l2": [],  # New
            "norm_scaling_cos": [],  # New
            "mapped_scaling_l2": [],  # New
            "mapped_scaling_cos": []   # New
        }


    def get_attack_direction(self, model, train_loader, max_retries=50, initial_lr=0.001, num_epochs=2):
        def contains_nan(tensor_list):
            return any(param is not None and torch.isnan(param).any() for param in tensor_list)

        model = deepcopy(model).to(self.device)
        model.train()
        loss_fn = lambda outputs, labels: -F.cross_entropy(outputs, labels)

        # Save original model state
        original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        retry_count = 0
        current_lr = initial_lr
        optimizer = optim.SGD(model.parameters(), lr=current_lr)

        while retry_count < max_retries:
            retry_count += 1
            model.load_state_dict({k: v.to(self.device) for k, v in original_state.items()})
            nan_detected = False

            for epoch in range(num_epochs):  # ✅ Run for multiple epochs
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()

                    if contains_nan([p.grad for p in model.parameters()]):
                        nan_detected = True
                        break

                    optimizer.step()

                    if contains_nan([p.data for p in model.parameters()]):
                        nan_detected = True
                        break

                if nan_detected:
                    break  # Exit epoch loop if NaN detected

            if nan_detected:
                current_lr *= 0.1
                optimizer = optim.SGD(model.parameters(), lr=current_lr)
                print(f"[Retry {retry_count}] NaN detected. Reducing lr to {current_lr} and restarting.")
            else:
                break  # ✅ Only break if all epochs completed without NaN

        if retry_count == max_retries:
            print("Maximum retries reached. Proceeding with last attempted state.")

        # Compute parameter differences as gradients
        grads = {}
        final_state = model.state_dict()
        for name in original_state:
            final_tensor = final_state[name].cpu()
            if torch.isnan(final_tensor).any():
                print(f"Warning: NaN in parameter {name} after training. Using zero change.")
                grads[name] = torch.zeros_like(original_state[name])
            else:
                grads[name] = final_tensor - original_state[name]

        # Flatten
        grad_flattened = [g.view(-1) for g in grads.values()]
        g_m_vec = torch.cat(grad_flattened)
        g_m_unit = g_m_vec / (g_m_vec.norm() + 1e-10)
        self.att_direction = g_m_unit

        # Restore original model state
        model.load_state_dict({k: v.to(self.device) for k, v in original_state.items()})

        grads = {k: v.to(self.device) for k, v in grads.items()}
        g_m_unit = g_m_unit.to(self.device)

        return grads, g_m_unit



    def compute_predicted_benign_update(self, model_t, model_t_minus_1, train_loader):
        # Simulate benign update
        # Direction using one batch of benign data
        # Norm using norm of server model delta
        model = deepcopy(model_t).to(self.device)
        model.eval()
        model.zero_grad()

        ce_loss = torch.nn.CrossEntropyLoss()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            loss = ce_loss(outputs, labels)
            loss.backward()
            break  # One batch is enough

        # Create a state_dict-shaped gradient dict
        grad_dict = {}
        named_param_dict = dict(model.named_parameters())
        model_sd = model.state_dict()  # includes buffers

        for key in model_sd:
            if key in named_param_dict:
                param = named_param_dict[key]
                if param.grad is not None:
                    grad_dict[key] = param.grad.detach().clone()
                else:
                    grad_dict[key] = torch.zeros_like(param)
            else:
                # For buffers like running_mean, running_var
                grad_dict[key] = torch.zeros_like(model_sd[key])

        # Flatten the full gradient
        grad_flattened = flatten_model(grad_dict)

        # Compute norm of model delta (using full state_dict)
        delta = flatten_model(model_t.state_dict()) - flatten_model(model_t_minus_1.state_dict())
        update_norm = delta.norm()

        benign_direction = grad_flattened / (grad_flattened.norm() + 1e-10)
        benign_update = update_norm * benign_direction

        return benign_update


    def estimate_benign_clients(self, total_clients_num, malicious_clients_num, current_s_m):
        """
        Estimate the number of benign clients selected in the current round.

        Args:
            total_clients (int): Total number of clients (p).
            malicious_clients (int): Number of malicious clients (q).
            current_s_m (int): Number of malicious clients selected in the current round.

        Returns:
            s_b (float): Estimated number of benign clients selected.
            s_hat (float): Estimated total number of selected clients per round.
        """
        self.history_s_m.append(current_s_m)
        history_len = len(self.history_s_m)
        self.logger.info(f"self.history_s_m len: {history_len}")
        
        if history_len == 0:
            # Fallback: assume uniform selection
            avg_s_m = current_s_m
        else:
            avg_s_m = sum(self.history_s_m) / history_len

        s_hat = (total_clients_num / malicious_clients_num) * avg_s_m
        s_b = s_hat - current_s_m
        return s_b, s_hat

    def estimate_server_update(self, s_b, s_m, benign_update, malicious_update):
        total = s_b + s_m
        return (s_b / total) * benign_update + (s_m / total) * malicious_update

    def update_history(self, delta_s, delta_b, delta_s_pred, s_m, delta_m):
        if not hasattr(self, "history_server_updates"):
            self.history_server_updates = []
            self.history_pred_benign_updates = []
            self.history_pred_server_updates = []
            self.history_malicious_updates = []
            self.history_malicious_selection = []

        self.history_server_updates.append(delta_s)
        self.history_pred_benign_updates.append(delta_b)
        self.history_pred_server_updates.append(delta_s_pred)
        self.history_malicious_updates.append(delta_m)
        self.history_malicious_selection.append(s_m)

        # Keep history size under E
        if len(self.history_server_updates) > self.E:
            self.history_server_updates.pop(0)
            self.history_pred_benign_updates.pop(0)
            self.history_pred_server_updates.pop(0)
            self.history_malicious_updates.pop(0)
            self.history_malicious_selection.pop(0)


    def bayesian_inference_l2(self, d, use_average=True):
        """
        Perform Bayesian inference using L2 distances, with an option to use either
        the average over all history or just the most recent iteration.

        :param d: Dimensionality of the parameter space.
        :param use_average: Boolean indicating whether to use the average of all iterations (True) or just the most recent one (False).
        :return: Posterior probability and the mean log likelihood if applicable.
        """
        E = len(self.history_server_updates)
        if E == 0:
            self.logger.warning("[FedOda] No history found for Bayesian inference. Returning neutral posterior.")
            return 0.5, None

        d0_list = []
        d1_list = []
        posteriors = []
        log_likelihoods = []

        for i in range(E):
            delta_s = self.history_server_updates[i]
            delta_b = self.history_pred_benign_updates[i]
            delta_s_pred = self.history_pred_server_updates[i]
            delta_m = self.history_malicious_updates[i]

            # Calculate L2 distances
            d0, d1, d2, d3 = self.get_l2_dist(delta_s, delta_b, delta_s_pred, delta_m)
            
            # Log the L2 squared distances
            self.logger.info(
                f"[FedOda] Iter t-{i}, "
                f"L2²(Δs, Δb')={d0:.4f}, "
                f"L2²(Δs, Δs')={d1:.4f}, "
                f"L2²(Δs, Δm)={d2:.4f}, "
                f"L2²(Δb', Δm)={d3:.4f}"
            )
            
            d0_list.append(d0)
            d1_list.append(d1)

            # Compute variance estimate
            if len(d0_list) > 1:
                sigma2 = sum(d0_list) / (d * len(d0_list))
            else:
                sigma2 = 1e-4

            sigma2 = max(sigma2, 1e-4)

            # Calculate log-likelihood for current iteration
            l = (d0 - d1) / (2 * sigma2)
            log_likelihoods.append(l)

            # Compute posterior for the current iteration
            ell_clamped = max(min(l, 80), -80)
            posterior = 1 / (1 + math.exp(-ell_clamped))
            posteriors.append(posterior)

            self.logger.info(f"[FedOda] Iteration {i} log-likelihood = {l:.4f} (clamped: {ell_clamped:.2f}), Posterior P(H1|D) = {posterior:.4f}")

        if use_average:
            # Use the average posterior of the iterations
            avg_posterior = sum(posteriors) / len(posteriors)
            self.logger.info(f"[FedOda] Average Posterior P(H1|D) = {avg_posterior:.4f}")
            return avg_posterior, sum(log_likelihoods) / len(log_likelihoods)
        else:
            # Use only the most recent iteration's posterior
            recent_posterior = posteriors[-1]
            self.logger.info(f"[FedOda] Using Recent Posterior P(H1|D) = {recent_posterior:.4f}")
            return recent_posterior, log_likelihoods[-1]



    def bayesian_inference_cosine(self):
        E = len(self.history_server_updates)
        if E == 0:
            self.logger.warning("[FedOda] No history found for Bayesian inference. Returning neutral posterior.")
            return 0.5

        d0_list = []
        d1_list = []
        log_likelihoods = []

        for i in range(E):
            delta_s = self.history_server_updates[i]
            delta_b = self.history_pred_benign_updates[i]
            delta_s_pred = self.history_pred_server_updates[i]
            delta_m = self.history_malicious_updates[i]

            # Replaced with log_cosine_dist call
            d0, d1, d2, d3 = self.get_cos_dist(delta_s, delta_b, delta_s_pred, delta_m)
            
            d0_list.append(d0)
            d1_list.append(d1)

            self.logger.info(f"[FedOda] Iter t-{i}")

        # ✅ Fix variance computation
        if len(d0_list) > 1:
            sigma2 = torch.var(torch.tensor(d0_list), unbiased=True).item()
        else:
            sigma2 = 1e-4  # fallback for E == 1

        sigma2 = max(sigma2, 1e-4)  # ensure nonzero

        self.logger.info(f"[FedOda] Estimated σ² = {sigma2:.6f}")

        for d0, d1 in zip(d0_list, d1_list):
            l = ((d0**2) - (d1**2)) / (2 * sigma2)
            log_likelihoods.append(l)

        # Final aggregation
        ell_total = sum(log_likelihoods)
        mean_log_likelihood = ell_total/len(log_likelihoods)
        
        # Clamp to prevent overflow in exp
        ell_total_clamped = max(min(ell_total, 80), -80)

        posterior = 1 / (1 + math.exp(-ell_total_clamped))

        self.logger.info(f"[FedOda] Cosine-based log-likelihood total = {ell_total:.4f} (clamped: {ell_total_clamped:.2f})")
        self.logger.info(f"[FedOda] Posterior P(H1|D) = {posterior:.4f}")

        return posterior, mean_log_likelihood


    def compute_attack_influence_r1(self, curr_acc, target_acc=0.2, tolerance=0.05):
        """
        r1 consists of:
        - Attack success (low accuracy)
        - Maintenance reward (staying low)
        - No reward when above threshold
        """
        # Normalize reward if under target
        attack_success = max(0.0, (target_acc - curr_acc) / target_acc)

        # Maintenance reward
        if curr_acc <= target_acc + tolerance:
            maintenance_bonus = 0.2  # small constant bonus
        else:
            maintenance_bonus = 0.0

        r1 = attack_success + maintenance_bonus
        return min(r1, 1.0)  # cap to 1.0


    # def compute_attack_influence_r2(self, prev_acc, curr_acc, target_acc=0.1, tolerance=0.03,
    #                             max_penalty=20.0, scale=50.0):
    #     if prev_acc is None:
    #         return 0.0
            
    #     delta = prev_acc - curr_acc
        
    #     # Case 1: Accuracy above target + tolerance → aggressive attack
    #     if curr_acc > target_acc + tolerance:
    #         return delta * scale  # Positive if accuracy dropped
            
    #     # Case 2: Within target range → maintenance
    #     elif curr_acc >= target_acc - tolerance:
    #         if delta > 0:  # Accuracy dropped
    #             return delta * (scale * 0.5)  # Smaller reward for maintenance
    #         else:  # Accuracy increased
    #             return delta * max_penalty  # Penalize increases
                
    #     # Case 3: Below target → small positive reward
    #     else:
    #         return 0.1  # Small constant reward for staying below target

    def compute_attack_influence_r2(self, prev_acc, curr_acc, scale=50.0):
        return (prev_acc - curr_acc) * scale / 100.0
    
    def compute_attack_influence_r3(self, scale=10.0, clip=True):
        """
        Compute r3 = (cos_b_dist - cos_m_dist) / (cos_b_dist + cos_m_dist), scaled to [-scale, scale].
        Args:
            scale: Reward scaling factor (default: 10.0).
            clip: Whether to enforce hard bounds (recommended).
        Returns:
            r3: Scaled reward in [-scale, scale], or None if missing data.
        """
        if (self.prev_server_model is None or
            self.prev_benign_update is None or
            self.prev_malicious_update is None):
            self.logger.warning("[FedOda] Missing history to compute r3. Returning None.")
            return None

        delta_s = flatten_model(self.server_model.state_dict()) - flatten_model(self.prev_server_model.state_dict())
        delta_b = self.prev_benign_update
        delta_m = self.prev_malicious_update

        cos_b = F.cosine_similarity(delta_b.unsqueeze(0), delta_s.unsqueeze(0), dim=1).item()
        cos_m = F.cosine_similarity(delta_m.unsqueeze(0), delta_s.unsqueeze(0), dim=1).item()

        cos_b_dist = 1 - cos_b  # Convert similarity to distance [0, 2]
        cos_m_dist = 1 - cos_m

        numerator = cos_b_dist - cos_m_dist
        denominator = cos_b_dist + cos_m_dist + 1e-8
        r3 = numerator / denominator  # Original r3 ∈ [-1, 1]

        # Scale and clip
        scaled_r3 = r3 * scale
        if clip:
            scaled_r3 = np.clip(scaled_r3, -scale, scale)

        self.logger.info(
            f"[FedOda] r3: cos_b_dist = {cos_b_dist:.4f}, cos_m_dist = {cos_m_dist:.4f}, "
            f"scaled_r3 = {scaled_r3:.4f}"
        )
        return float(scaled_r3)

    def compare_proj_vs_pgd_attack(self, benign_update, target_direction, scaling_cos, epsilon):
        """
        Compare accuracy after applying PROJ vs PGD malicious updates.
        """
        device = self.device
        model = self.server_model
        original_model_state = deepcopy(model.state_dict())
        original_acc = self.validate(model, self.pooled_balanced_loader)

        print(f"Original Accuracy: {original_acc:.4f}")

        # === PROJ mode
        model.load_state_dict(original_model_state)
        delta_proj = self.mali_update_craft_l2_cos(
            model=deepcopy(model),
            benign_update=benign_update,
            target_direction=target_direction,
            scaling_cos_dist=scaling_cos,
            epsilon=epsilon,
            mode="PROJ"
        )
        model_proj = self.apply_update_to_model(model, delta_proj)
        acc_proj = self.validate(model_proj, self.pooled_balanced_loader)
        acc_drop_proj = original_acc - acc_proj

        print(f"[PROJ] Accuracy After Attack: {acc_proj:.4f} | ΔAcc: {acc_drop_proj:.4f} | ||Δ|| = {delta_proj.norm():.4f}")

        # === PGD mode
        model.load_state_dict(original_model_state)
        delta_pgd = self.mali_update_craft_l2_cos(
            model=deepcopy(model),
            benign_update=benign_update,
            target_direction=target_direction,
            scaling_cos_dist=scaling_cos,
            epsilon=epsilon,
            mode="PGD",
            pgd_steps=20,
            pgd_lr=0.1
        )
        model_pgd = self.apply_update_to_model(model, delta_pgd)
        acc_pgd = self.validate(model_pgd, self.pooled_balanced_loader)
        acc_drop_pgd = original_acc - acc_pgd

        print(f"[PGD ] Accuracy After Attack: {acc_pgd:.4f} | ΔAcc: {acc_drop_pgd:.4f} | ||Δ|| = {delta_pgd.norm():.4f}")

        # === Comparison
        if acc_pgd < acc_proj:
            print("✅ PGD is more harmful than PROJ under same constraint.")
        elif acc_pgd > acc_proj:
            print("⚠️ PGD is less harmful than PROJ in this trial.")
        else:
            print("ℹ️ PGD and PROJ are equally harmful.")

        return {
            "original_acc": original_acc,
            "proj_acc": acc_proj,
            "pgd_acc": acc_pgd,
            "proj_delta": delta_proj,
            "pgd_delta": delta_pgd
        }


    # def mali_update_craft_l2_cos(
    #     self,
    #     model,
    #     benign_update,
    #     target_direction,
    #     scaling_cos_dist,
    #     epsilon,
    #     harm_threshold=0.0,
    #     retry_limit=5,
    #     epsilon_growth=1,
    #     cos_growth=0.9,
    # ):
    #     """
    #     Craft a malicious update Δₘ such that:
    #         - Lies in the plane of `target_direction` and `benign_update`
    #         - L2 norm ≈ epsilon (relaxed over retries)
    #         - Cosine distance(Δₘ, target_direction) ≈ scaling_cos_dist (relaxed over retries)
    #         - Reduces validation accuracy

    #     If no such update is found, returns best effort or zero update if all increased accuracy.

    #     Args:
    #         model (nn.Module): Server model to attack.
    #         benign_update (torch.Tensor): Benign model update.
    #         target_direction (torch.Tensor): Malicious direction (unit vector).
    #         scaling_cos_dist (float): Desired cosine distance from target_direction.
    #         epsilon (float): L2-norm constraint.
    #         harm_threshold (float): Minimum accuracy drop to consider harmful.
    #         retry_limit (int): Max attempts if update is not harmful enough.
    #         epsilon_growth (float): Factor to increase epsilon each retry.
    #         cos_growth (float): Factor to increase cosine distance each retry.

    #     Returns:
    #         torch.Tensor: Crafted malicious update Δₘ.
    #     """
    #     original_model_state = deepcopy(model.state_dict())
    #     original_acc = self.validate(model, self.pooled_balanced_loader)

    #     best_delta = None
    #     best_acc_drop = -float("inf")
    #     all_acc_improving = True

    #     for attempt in range(retry_limit):
    #         # === Step 1: Create direction with desired cosine distance
    #         crafted_direction = self.project_to_cosine(
    #             v=benign_update,
    #             ref=target_direction,
    #             desired_cos_dist=scaling_cos_dist
    #         )

    #         # === Step 2: Scale to L2 norm = epsilon
    #         delta = crafted_direction / crafted_direction.norm() * epsilon

    #         # === Step 3: Apply delta to model and evaluate
    #         attacked_model = self.apply_update_to_model(model, delta)
    #         new_acc = self.validate(attacked_model, self.pooled_balanced_loader)
    #         acc_drop = original_acc - new_acc

    #         cos_dist = 1 - F.cosine_similarity(
    #             delta.unsqueeze(0), target_direction.unsqueeze(0), dim=1
    #         ).item()

    #         self.logger.info(
    #             f"[FedOda][Craft] Attempt {attempt + 1}: Acc = {new_acc:.4f}, ΔAcc = {-acc_drop:.4f}, CosDist = {cos_dist:.4f}, ||Δ|| = {delta.norm():.4f}"
    #         )

    #         # === Check if this attempt is harmful enough
    #         if acc_drop >= 0:
    #             self.logger.info(f"[FedOda][Craft] Successful attack at attempt {attempt + 1}")
    #             model.load_state_dict(original_model_state)
    #             return delta.detach()

    #         # Track best effort
    #         if acc_drop > best_acc_drop:
    #             best_acc_drop = acc_drop
    #             best_delta = delta.detach()

    #         # If any attempt reduced accuracy, set flag
    #         if acc_drop < 0:
    #             all_acc_improving = False

    #         # === Step 4: Relax constraints and retry
    #         scaling_cos_dist *= cos_growth
    #         epsilon *= epsilon_growth
    #         model.load_state_dict(original_model_state)

    #     # === Final fallback
    #     if all_acc_improving:
    #         self.logger.warning("[FedOda][Craft] All attempted updates improved accuracy. Sending zero update.")
    #         model.load_state_dict(original_model_state)
    #         return torch.zeros_like(benign_update)

    #     self.logger.warning("[FedOda][Craft] No harmful delta found. Returning best effort - original server model.")
    #     model.load_state_dict(original_model_state)
    #     return best_delta


    def mali_update_craft_l2_cos(
        self,
        model,
        benign_update,
        target_direction,
        scaling_cos_dist,
        epsilon,
        harm_threshold=0.0,
        retry_limit=5,
        epsilon_growth=1,
        cos_growth=0.9,
        mode="PROJ",
        pgd_steps=20,
        pgd_lr=0.1,
    ):
        """
        Craft a malicious update Δₘ under constraints using either:
            - PROJ: project + rescale heuristic (default)
            - PGD: optimize for validation loss under constraints

        Returns:
            torch.Tensor: Malicious update
        """
        original_model_state = deepcopy(model.state_dict())
        original_acc = self.validate(model, self.pooled_balanced_loader)

        best_delta = None
        best_acc_drop = -float("inf")
        all_acc_improving = True

        for attempt in range(retry_limit):
            if mode == "PROJ":
                # === Step 1: Project to desired cosine distance
                crafted_direction = self.project_to_cosine(
                    v=benign_update,
                    ref=target_direction,
                    desired_cos_dist=scaling_cos_dist
                )
                # === Step 2: Scale to L2 norm = epsilon
                delta = crafted_direction / crafted_direction.norm() * epsilon
            elif mode == "PGD":
                delta = self.pgd_craft_attack_delta(
                    model=model,
                    target_direction=target_direction,
                    scaling_cos_dist=scaling_cos_dist,
                    epsilon=epsilon,
                    steps=pgd_steps,
                    lr=pgd_lr,
                    device=self.device
                )
            else:
                raise ValueError(f"Unknown attack mode: {mode}")

            # === Step 3: Apply and validate
            attacked_model = self.apply_update_to_model(model, delta)
            new_acc = self.validate(attacked_model, self.pooled_balanced_loader)
            acc_drop = original_acc - new_acc

            def safe_cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
                a_norm = a.norm()
                b_norm = b.norm()
                if a_norm < eps or b_norm < eps:
                    return float("nan")
                cos_sim = torch.dot(a, b) / (a_norm * b_norm)
                return 1 - cos_sim.item()

            cos_dist = safe_cosine_distance(delta, target_direction)

            cos_dist_target = safe_cosine_distance(delta, target_direction)
            cos_dist_benign = safe_cosine_distance(delta, benign_update)

            self.logger.info(
                f"[FedOda][Craft-{mode}] Attempt {attempt + 1}: Acc = {new_acc:.4f}, ΔAcc = {-acc_drop:.4f}, "
                f"CosDist Δ_m~Δ_t = {cos_dist_target:.4f} | Δ_m~Δ_b = {cos_dist_benign:.4f} | ||Δ|| = {delta.norm():.4f}"
            )

            self.logger.info(
                f"[FedOda][Craft-{mode}] Attempt {attempt + 1}: Acc = {new_acc:.4f}, ΔAcc = {-acc_drop:.4f}, CosDist = {cos_dist:.4f}, ||Δ|| = {delta.norm():.4f}"
            )

            # === Check for successful attack
            if acc_drop >= harm_threshold:
                self.logger.info(f"[FedOda][Craft-{mode}] Successful attack at attempt {attempt + 1}")
                model.load_state_dict(original_model_state)
                return delta.detach()

            # Track best effort
            if acc_drop > best_acc_drop:
                best_acc_drop = acc_drop
                best_delta = delta.detach()

            if acc_drop < 0:
                all_acc_improving = False

            # === Step 4: Relax constraints
            scaling_cos_dist *= cos_growth
            epsilon *= epsilon_growth
            model.load_state_dict(original_model_state)

        # === Final fallback
        if all_acc_improving:
            self.logger.warning(f"[FedOda][Craft-{mode}] All updates improved accuracy. Sending zero update.")
            model.load_state_dict(original_model_state)
            return torch.zeros_like(benign_update)

        self.logger.warning(f"[FedOda][Craft-{mode}] No harmful update found. Returning best effort.")
        model.load_state_dict(original_model_state)
        return best_delta

    def pgd_craft_attack_delta(
        self,
        model,
        target_direction,
        scaling_cos_dist,
        epsilon,
        steps=20,
        lr=0.1,
        device=None
    ):
        """
        PGD attack update to increase validation loss under:
            - L2 norm == ε
            - Cosine distance >= scaling_cos_dist
        """
        model_flat = self.get_flat_model_params(model)
        delta = torch.zeros_like(model_flat, requires_grad=True)

        optimizer = torch.optim.SGD([delta], lr=lr)

        # Normalize the target direction once
        target_direction = target_direction / (target_direction.norm() + 1e-8)

        for step in range(steps):
            optimizer.zero_grad()

            # Apply delta to model
            model_copy = deepcopy(model)
            self.set_flat_model_params(model_copy, model_flat + delta)

            # Get one batch
            x, y = next(iter(self.pooled_balanced_loader))
            x, y = x.to(device), y.to(device)

            model_copy.train()
            logits = model_copy(x)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, y)

            (-loss).backward()
            optimizer.step()

            with torch.no_grad():
                # === Optional: Enforce cosine constraint only if violated
                current_cos_sim = torch.dot(delta, target_direction) / (
                    delta.norm() * target_direction.norm() + 1e-8
                )
                current_cos_dist = 1 - current_cos_sim.item()

                if current_cos_dist < scaling_cos_dist:
                    # Project to desired cosine constraint
                    delta.data = self.project_to_cosine(
                        v=delta.data, ref=target_direction, desired_cos_dist=scaling_cos_dist
                    )

                # Rescale to exact L2 norm = epsilon
                if delta.norm() > 1e-6:
                    delta.data = delta.data / delta.data.norm() * epsilon
                else:
                    # Avoid dividing by 0
                    delta.data = torch.randn_like(delta.data)
                    delta.data = delta.data / delta.data.norm() * epsilon

        return delta.detach()


    def get_flat_model_params(self, model):
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    def set_flat_model_params(self, model, flat_tensor):
        pointer = 0
        for p in model.parameters():
            num_params = p.numel()
            new_data = flat_tensor[pointer:pointer + num_params].view_as(p)
            p.data.copy_(new_data)
            pointer += num_params


    def get_cos_dist(self, delta_s, delta_b, delta_s_pred, delta_m):
        # Calculate cosine distances
        d1 = 1 - F.cosine_similarity(delta_s, delta_b, dim=0).item()
        d2 = 1 - F.cosine_similarity(delta_s, delta_s_pred, dim=0).item()
        d3 = 1 - F.cosine_similarity(delta_s, delta_m, dim=0).item()
        d4 = 1 - F.cosine_similarity(delta_b, delta_m, dim=0).item()

        return d1, d2, d3, d4


    def get_l2_dist(self, delta_s, delta_b, delta_s_pred, delta_m):
        # Calculate L2 squared distances
        d0 = (delta_s - delta_b).norm()**2           # ||Δs - Δb'||²
        d1 = (delta_s - delta_s_pred).norm()**2      # ||Δs - Δs'||²
        d2 = (delta_s - delta_m).norm()**2           # ||Δs - Δm||²
        d3 = (delta_b - delta_m).norm()**2           # ||Δb' - Δm||²

        return d0, d1, d2, d3


    def sign_based_bayesian_inference(self, epsilon=1e-6):
        E = len(self.history_server_updates)
        if E == 0:
            self.logger.warning("[FedOda] No history available for sign-based inference.")
            return 0.5

        delta_s = self.history_server_updates[-1]
        delta_b = self.history_pred_benign_updates[-1]
        delta_m = self.history_malicious_updates[-1]

        sign_b = torch.sign(delta_b)
        sign_m = torch.sign(delta_m)
        sign_s = torch.sign(delta_s)

        # Only consider conflicting coordinates (Δb ≠ Δm) and non-zero signs
        conflict_mask = (sign_b != sign_m) & (sign_b != 0) & (sign_m != 0)

        if conflict_mask.sum().item() == 0:
            self.logger.warning("[FedOda] No conflicting coordinates found between benign and malicious signs.")
            return 0.5

        total_conflict = conflict_mask.sum().item()

        agree_m = ((sign_s == sign_m) & conflict_mask).sum().item()
        agree_b = ((sign_s == sign_b) & conflict_mask).sum().item()

        agree_rate_m = agree_m / total_conflict
        agree_rate_b = agree_b / total_conflict

        log_likelihood_ratio = math.log((agree_rate_m + epsilon) / (agree_rate_b + epsilon))
        posterior = 1 / (1 + math.exp(-log_likelihood_ratio))

        self.logger.info(
            f"[FedOda] Sign-based Bayesian Inference: "
            f"total_dims={delta_s.numel()}, conflict={total_conflict} "
            f"({total_conflict / delta_s.numel():.2f}%), "
            f"agree_m={agree_m}, agree_b={agree_b}, "
            f"rate_m={agree_rate_m:.4f}, rate_b={agree_rate_b:.4f}, "
            f"log_likelihood_ratio={log_likelihood_ratio:.4f}, "
            f"posterior={posterior:.4f}"
        )

        return posterior



    def adjust_scaling_factor(self, scaling_factor, exploration_prob, exploration_scale):
        """
        Adjusts the scaling factor using an exponentially weighted average of posterior history,
        and exploration to avoid local optima.
        """
        if not self.posterior_history or all(p is None for p in self.posterior_history):
            return scaling_factor  # fallback if no valid data

        # Use most recent valid posterior
        recent_posterior = next((p for p in reversed(self.posterior_history) if p is not None), None)
        if recent_posterior is None:
            return scaling_factor  # fallback

        # Update EMA
        if self.ema_posterior is None:
            self.ema_posterior = recent_posterior
        else:
            self.ema_posterior = self.ema_alpha * recent_posterior + (1 - self.ema_alpha) * self.ema_posterior

        avg_posterior = self.ema_posterior

        # Log full history
        formatted_history = [f"{p:.4f}" if p is not None else "None" for p in self.posterior_history]
        self.logger.info(f"[FedOda] Posterior history (last {len(self.posterior_history)} rounds): {formatted_history}")
        self.logger.info(f"[FedOda] EMA posterior = {self.ema_posterior:.4f}")

        # Parameters
        margin = 0.1
        lower = self.threshold - margin
        upper = self.threshold + margin

        # exploration_prob = 0.1
        # exploration_scale = 1.5

        should_explore = random.random() < exploration_prob

        if should_explore:
            direction = random.choice(['up', 'down'])
            if direction == 'up':
                new_factor = scaling_factor * exploration_scale
                self.logger.info("[FedOda] Exploration: Trying increased scaling factor.")
            else:
                new_factor = scaling_factor * self.reduce_factor
                self.logger.info("[FedOda] Exploration: Trying reduced scaling factor.")
        else:
            if avg_posterior < lower:
                new_factor = scaling_factor * self.reduce_factor
                self.logger.info(f"[FedOda] EMA posterior low < {lower:.4f} — reducing scaling.")
            elif avg_posterior > upper:
                new_factor = scaling_factor * self.increase_factor
                self.logger.info(f"[FedOda] EMA posterior high > {upper:.4f} — increasing scaling.")
            else:
                new_factor = scaling_factor * 1.05
                self.logger.info(f"[FedOda] EMA posterior near threshold — slightly increasing.")

        new_factor = max(0.5, min(20.0, new_factor))
        return new_factor


    def simulate_benign_direction(self, loader, data_size=500, batch_size=64):
        """
        Simulate benign training to estimate the update norm for one client.

        Args:
            loader (DataLoader): DataLoader containing (input, label) batches.
            data_size (int): Number of data points to use.
            batch_size (int): Mini-batch size for training.

        Returns:
            update_vector (Tensor): Flattened parameter update vector.
            update_norm (float): L2 norm of the update vector.
        """

        model_before = deepcopy(self.server_model)
        model_after = deepcopy(self.server_model).to(self.device)
        model_after.train()
        optimizer = torch.optim.SGD(model_after.parameters(), lr=self.client_learning_rate)

        # Collect up to data_size samples from the loader
        collected_x, collected_y = [], []
        total = 0
        for inputs, labels in loader:
            collected_x.append(inputs)
            collected_y.append(labels)
            total += inputs.size(0)
            if total >= data_size:
                break

        if total == 0:
            raise ValueError("simulate_benign_update(): No data found in DataLoader.")

        # Concatenate and truncate to exactly data_size
        inputs = torch.cat(collected_x, dim=0)[:data_size]
        labels = torch.cat(collected_y, dim=0)[:data_size]

        # Create new DataLoader from collected data
        temp_loader = DataLoader(list(zip(inputs, labels)), batch_size=batch_size, shuffle=True, pin_memory=False)

        # Simulate local training
        for batch_inputs, batch_labels in temp_loader:
            batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
            optimizer.zero_grad()
            outputs = model_after(batch_inputs)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Compute the update
        before = flatten_model(model_before.state_dict())
        after = flatten_model(model_after.state_dict())
        update = after - before
        norm = update.norm()
        update_direction = update / (norm + 1e-10)  # Normalize to unit vector

        return update_direction, norm.item()


    def simulate_benign_updates_l2_cos(self, sample_num, batch_size, mco_dict, c_epochs=1):
        """
        Simulate benign training for a random sample of malicious clients, calculate distances,
        and restore original weights (non-destructive operation).
        
        Args:
            sample_num: Number of malicious clients to randomly sample
            batch_size: Batch size for training
            mco_dict: Dictionary containing client models and optimizers
            c_epochs: Number of training epochs
            
        Returns:
            avg_l2_distance: Average L2 distance between all pairs of clients
            avg_cos_distance: Average cosine distance between all pairs of clients
            avg_update_norm: Average L2 norm of individual client updates
            updates_dict: Dictionary containing all client updates {client_id: update}
            sampled_clients: List of client IDs that were sampled
        """
        if len(self.malicious_client_ids) < 2:
            raise ValueError("Need at least 2 malicious clients to calculate distances")
        
        # Randomly sample malicious clients
        sample_num = min(sample_num, len(self.malicious_client_ids))
        sampled_clients = random.sample(self.malicious_client_ids, sample_num)
        
        updates_dict = {}
        original_states = {}  # Store original model states
        update_norms = []     # Store L2 norms of individual updates
        full_model_cos_dists = []
        
        for client_id in sampled_clients:
            # Store original state
            original_states[client_id] = deepcopy(mco_dict[client_id]["c_model"].state_dict())
            
            # Get client-specific data
            client_data = self.fl_mali_data_dict[client_id]
            client_dl = DataLoader(client_data, batch_size=batch_size, 
                                shuffle=True, num_workers=1, drop_last=True, pin_memory=False)
            
            # Train the model on client's own data
            mco_dict[client_id]["c_model"].train()
            
            for epoch in range(c_epochs):
                for batch_idx, (data, target) in enumerate(client_dl):
                    data, target = data.to(self.device), target.to(self.device)
                    mco_dict[client_id]["c_optimizer"].zero_grad()
                    output = mco_dict[client_id]["c_model"](data)
                    loss = mco_dict[client_id]["c_criterion"](output, target)
                    loss.backward()
                    mco_dict[client_id]["c_optimizer"].step()
            
            # Calculate update (difference between original and trained model)
            update = {}
            for key in original_states[client_id]:
                update[key] = (mco_dict[client_id]["c_model"].state_dict()[key] - 
                            original_states[client_id][key])
            
            # Calculate and store L2 norm of this update
            flattened_update = flatten_model(update)
            update_norm = torch.norm(flattened_update, p=2).item()
            update_norms.append(update_norm)
            
            updates_dict[client_id] = update
        
        # Calculate pairwise distances
        l2_distances = []
        cos_distances = []
        client_pairs = list(itertools.combinations(sampled_clients, 2))
        
        for (client_i, client_j) in client_pairs:
            # Calculate L2 distance
            l2_dist = self._calculate_update_distance(
                updates_dict[client_i], 
                updates_dict[client_j], 
                distance_type='l2'
            )
            l2_distances.append(l2_dist)
            
            # Calculate cosine distance
            cos_dist = self._calculate_update_distance(
                updates_dict[client_i], 
                updates_dict[client_j], 
                distance_type='cos'
            )
            cos_distances.append(cos_dist)

            # Now compute cosine distance between (update + server state)
            model_i = deepcopy(original_states[client_i])
            model_j = deepcopy(original_states[client_j])

            # Add update to server model (non-destructive)
            for k in model_i:
                model_i[k] = model_i[k] + updates_dict[client_i][k]
                model_j[k] = model_j[k] + updates_dict[client_j][k]

            cos_full_model = self._calculate_update_distance(model_i, model_j, distance_type='cos')
            full_model_cos_dists.append(cos_full_model)

        
        # Restore original model states
        for client_id in sampled_clients:
            mco_dict[client_id]["c_model"].load_state_dict(original_states[client_id])
        
        # Calculate averages
        avg_l2_distance = sum(l2_distances) / len(l2_distances) if l2_distances else 0
        avg_cos_distance = sum(cos_distances) / len(cos_distances) if cos_distances else 0
        avg_update_norm = sum(update_norms) / len(update_norms) if update_norms else 0
        avg_full_model_cos = sum(full_model_cos_dists) / len(full_model_cos_dists) if full_model_cos_dists else 0
        
        return avg_l2_distance, avg_cos_distance, avg_update_norm, avg_full_model_cos


    def craft_malicious_update_cos_budget(self, Delta_m, Delta_b, epsilon_cos):
        """
        Crafts Delta_m' such that cos_distance(Delta_m', Delta_b) <= epsilon_cos.
        Inputs are normalized to isolate directional changes.
        
        Args:
            Delta_m (dict/torch.Tensor): Original malicious update (state_dict or tensor).
            Delta_b (dict/torch.Tensor): Benign update (same format as Delta_m).
            epsilon_cos (float): Max allowed cosine distance (0 ≤ epsilon_cos ≤ 2).
            
        Returns:
            Delta_m_prime (same format as input): Crafted update satisfying the constraint.
        """
        # Convert inputs to flattened tensors if they are state_dicts
        def _to_tensor(x):
            if isinstance(x, dict):
                return torch.cat([v.flatten() for v in x.values()])
            return x.flatten()
        
        Delta_m_flat = _to_tensor(Delta_m)
        Delta_b_flat = _to_tensor(Delta_b)
        
        # Normalize to unit vectors
        Delta_m_unit = Delta_m_flat / torch.norm(Delta_m_flat)
        Delta_b_unit = Delta_b_flat / torch.norm(Delta_b_flat)
        
        # First check if Delta_m already satisfies the constraint
        cos_sim_initial = F.cosine_similarity(
            Delta_m_unit.unsqueeze(0), 
            Delta_b_unit.unsqueeze(0)
        ).item()
        cos_dist_initial = 1 - cos_sim_initial
        
        if cos_dist_initial <= epsilon_cos:
            self.logger.info(f"[FedOda] Original update already satisfies constraint (cos_dist={cos_dist_initial:.4f} ≤ {epsilon_cos:.4f})")
            return Delta_m
        
        # Binary search for alpha (mixing factor)
        low, high = 0.0, 1.0
        best_alpha = 0.0  # Default: fully move to Delta_b
        
        for _ in range(100):  # Sufficient iterations
            alpha = (low + high) / 2
            Delta_m_prime_unit = alpha * Delta_m_unit + (1 - alpha) * Delta_b_unit
            Delta_m_prime_unit /= torch.norm(Delta_m_prime_unit)  # Re-normalize
            
            # Calculate cosine distance with Delta_b
            cos_sim = F.cosine_similarity(
                Delta_m_prime_unit.unsqueeze(0), 
                Delta_b_unit.unsqueeze(0)
            ).item()
            cos_dist = 1 - cos_sim
            
            if cos_dist <= epsilon_cos:
                best_alpha = alpha
                low = alpha  # Try to stay closer to Delta_m
            else:
                high = alpha  # Move toward Delta_b
        
        self.logger.info(f"[FedOda] Best alpha found: {best_alpha:.4f} for epsilon_cos={epsilon_cos:.4f}")
        cos_dist = 1 - F.cosine_similarity(
            (best_alpha * Delta_m_unit + (1 - best_alpha) * Delta_b_unit).unsqueeze(0),
            Delta_b_unit.unsqueeze(0)
        ).item()
        self.logger.info(f"[FedOda] Cosine distance to benign update: {cos_dist:.4f}")
        
        # Compute final Delta_m_prime (restore original norm)
        Delta_m_prime_flat = best_alpha * Delta_m_unit + (1 - best_alpha) * Delta_b_unit
        
        # Convert back to original format
        if isinstance(Delta_m, dict):
            Delta_m_prime = {}
            ptr = 0
            for k, v in Delta_m.items():
                numel = v.numel()
                Delta_m_prime[k] = Delta_m_prime_flat[ptr:ptr+numel].reshape_as(v)
                ptr += numel
        else:
            Delta_m_prime = Delta_m_prime_flat.reshape_as(Delta_m)
        
        return Delta_m_prime


    def _calculate_update_distance(self, update1, update2, distance_type='l2'):
        """
        Calculate distance between two updates.
        
        Args:
            update1: First model update (state_dict)
            update2: Second model update (state_dict)
            distance_type: 'l2' for L2 distance, 'cos' for cosine distance
            
        Returns:
            distance: Calculated distance
        """
        if distance_type not in ['l2', 'cos']:
            raise ValueError("distance_type must be either 'l2' or 'cos'")
            
        # Flatten updates into vectors
        vec1 = []
        vec2 = []
        for key in update1:
            vec1.append(update1[key].view(-1))
            vec2.append(update2[key].view(-1))
        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)
        
        if distance_type == 'l2':
            # Calculate L2 distance
            distance = torch.norm(vec1 - vec2, p=2).item()
        else:
            # Calculate cosine distance (1 - cosine similarity)
            cos_sim = torch.nn.functional.cosine_similarity(
                vec1.unsqueeze(0), vec2.unsqueeze(0)
            ).item()
            distance = 1 - cos_sim
            
        return distance

    def validate(self, model=None, loader=None):
        """
        Evaluate the given model (or server_model by default) on the given loader 
        (or self.pooled_loader by default) and return validation accuracy.

        Args:
            model: The model to evaluate. If None, uses self.server_model.
            loader: Dataloader for evaluation. If None, uses self.pooled_loader.

        Returns:
            val_accuracy (float): Accuracy in percentage (0–100)
        """
        if loader is None:
            loader = self.pooled_loader

        if loader is None:
            self.logger.warning("[FedOda] No validation loader provided - skipping validation")
            return None

        if model is None:
            model = self.server_model

        # Create a copy of the model to avoid modifying the original
        model_copy = self.model_fn().to(self.device)
        model_copy.load_state_dict(model.state_dict())

        val_accuracy = self._evaluate_model(model_copy, loader)

        return val_accuracy


    def apply_update_to_model(self, model, update_vector):
        """
        Apply a flat update vector to a model.

        Args:
            model (nn.Module): Model to update.
            update_vector (torch.Tensor): Flattened update.

        Returns:
            nn.Module: Updated model.
        """
        
        new_model = deepcopy(model)
        update_dict = vector_to_state_dict(update_vector, new_model)
        # for (name, param), (_, update) in zip(new_model.named_parameters(), update_dict.items()):
        #     # print(f"[DEBUG] update.dtype = {update.dtype}, param dtype = {param.data.dtype}")
        #     param.data += update

        for name, param in new_model.named_parameters():
            if name in update_dict:
                update = update_dict[name]
                if param.data.shape == update.shape:  # Ensure shapes match
                    param.data += update.to(param.data.dtype)  # Ensure dtype compatibility
                else:
                    raise ValueError(f"Shape mismatch for {name}: model param {param.data.shape} vs update {update.shape}")


        return new_model


    def compute_loss(self, model, data_loader, require_grad=False):
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        if not require_grad:
            torch.set_grad_enabled(False)

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            loss = loss_fn(logits, y)

            if require_grad:
                total_loss += loss * x.size(0)
            else:
                total_loss += loss.item() * x.size(0)

            total_samples += x.size(0)

        if not require_grad:
            torch.set_grad_enabled(True)

        if require_grad:
            avg_loss = total_loss / total_samples
        else:
            avg_loss = total_loss / total_samples

        return avg_loss


    def project_to_cosine(self, v, ref, desired_cos_dist):
        """
        Project vector `v` onto the 2D plane spanned by `v` and `ref`,
        such that it has cosine distance = desired_cos_dist to `ref`.

        Args:
            v (torch.Tensor): Vector to be projected (e.g., base_update).
            ref (torch.Tensor): Reference direction (e.g., target_direction).
            desired_cos_dist (float): Target cosine distance to `ref`.

        Returns:
            torch.Tensor: Projected vector.
        """
        desired_cos_sim = 1.0 - desired_cos_dist  # Convert distance to similarity

        # ✅ Clamp to avoid numerical errors that result in complex numbers
        desired_cos_sim = max(min(desired_cos_sim, 1.0), -1.0)

        ref_unit = ref / (ref.norm() + 1e-10)
        v_proj = v - (v @ ref_unit) * ref_unit
        v_proj_norm = v_proj.norm() + 1e-10

        a = desired_cos_sim
        b_squared = 1.0 - desired_cos_sim ** 2

        # ✅ Prevent negative sqrt due to floating point error
        b = (b_squared if b_squared > 0 else 0.0) ** 0.5

        v_proj_unit = v_proj / v_proj_norm
        projected_direction = a * ref_unit + b * v_proj_unit
        return projected_direction




    def _evaluate_model(self, model, loader):
        """
        Evaluate the given model on the given data loader.

        Args:
            model: The model to evaluate.
            loader: Data loader for evaluation.

        Returns:
            val_accuracy (float): Accuracy in percentage (0–100)
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total



    def _evaluate_malicious_update(self, malicious_update):
        """
        Evaluate the impact of a malicious update by applying it to the server model
        and measuring validation accuracy.
        
        Args:
            malicious_update: The malicious update vector to evaluate
            
        Returns:
            val_accuracy: Validation accuracy of the malicious model (0-100)
        """
        if self.vali_loader is None:
            self.logger.warning("[FedOda] No validation loader provided - skipping evaluation")
            return None
        
        # Create a copy of the server model
        malicious_model = self.model_fn().to(self.device)
        malicious_model.load_state_dict(self.server_model.state_dict())
        
        # Apply the malicious update
        with torch.no_grad():
            for param, update in zip(malicious_model.parameters(), malicious_update):
                param.data += update
        
        # Evaluate the model
        val_accuracy = self._evaluate_model(malicious_model, self.vali_loader)
        
        return val_accuracy


    def attack(self, sample_num=3, batch_size=64, mco_dict=None, c_epochs=1):
        if self.server_model is None:
            self.logger.warning("[FedOda] Server model is None. Cannot proceed with attack.")
            return {}, {}

        # === Update server model
        self.previous_server_model = deepcopy(self.server_model)
        self.server_model = deepcopy(mco_dict[-1]["c_model"])

        # === Validate server model (always once per round)
        acc_curr = self.validate(self.server_model, self.pooled_balanced_loader)
        self.val_acc_pooled_history.append(acc_curr)

        round_number = len(self.val_acc_pooled_history)
        acc_prev = self.val_acc_pooled_history[-2] if round_number > 1 else acc_curr

        # === If no malicious clients → skip PPO and attack
        if self.current_s_m == 0:
            self.logger.info(f"[FedOda] No malicious clients selected in round {round_number}. Skipping attack.")
            return {}, {}

        # === Proceed with regular attack
        return self._regular_attack(sample_num, batch_size, mco_dict, c_epochs)





    def estimate_client_distribution(self):
        """
        Estimate the number of benign clients and compute the malicious client portion (alpha).
        Uses previous `s_m` if available; otherwise falls back to defaults.
        Updates internal state variables as needed.
        """
        if self.prev_s_m is not None and self.prev_s_m > 0:
            s_b, s_hat = self.estimate_benign_clients(
                total_clients_num=self.total_clients_num,
                malicious_clients_num=self.malicious_clients_num,
                current_s_m=self.prev_s_m
            )
        else:
            s_b = self.total_clients_num - self.malicious_clients_num
            s_hat = self.total_clients_num
            self.logger.warning("[FedOda] prev_s_m is None. Using fallback s_b and s_hat.")

        s_m = self.prev_s_m if self.prev_s_m else self.malicious_clients_num
        alpha = s_m / s_hat  # malicious client portion

        self.logger.info(f"[FedOda] Estimated s_b = {s_b:.2f}, s_hat = {s_hat:.2f}, s_m = {s_m}, alpha = {alpha:.4f}")

        return s_b, s_hat, s_m, alpha


    def compute_malicious_update_direction(s_m, s_hat, g_m_vec, benign_update, beta):
        """
        Compute the malicious update direction vector `v`.

        Args:
            s_m (float): Estimated number of malicious clients.
            s_hat (float): Estimated total number of participating clients.
            g_m_vec (Tensor): Malicious clients' aggregated update vector.
            benign_update (Tensor): Estimated benign update vector (Δ_b).
            beta (float): Scaling factor for the malicious direction.

        Returns:
            Tensor: Malicious update direction vector `v`.
        """
        alpha = s_m / (s_hat + 1e-10)  # Malicious client portion, with small epsilon for numerical stability
        v = (beta * g_m_vec) - ((1 - alpha) * benign_update)
        v = v / (alpha + 1e-10)
        return v


    def log_to_full_history(self, *,
                            s_b=None, s_hat=None,
                            benign_norm=None, server_update_norm=None,
                            posterior=None,
                            norm_scaling_l2=None, norm_scaling_cos=None,
                            mapped_scaling_l2=None, mapped_scaling_cos=None,
                            r1=None, r2=None, r3=None,
                            scalar_reward=None,
                            cos_delta=None, cos_b_b=None,
                            l2_delta=None, l2_b_b=None,
                            model_cos_delta=None, model_cos_b_b=None):
        """
        Logs a single iteration's key metrics to self.full_history.
        """

        if not hasattr(self, "full_history"):
            self.full_history = {}

        # Ensure all required keys exist
        keys_to_init = {
            "s_b": [], "s_hat": [],
            "benign_norm": [], "server_update_norm": [],
            "posterior": [],
            "scaling_factor": [], "epsilon": [],
            "norm_scaling_l2": [], "norm_scaling_cos": [],
            "mapped_scaling_l2": [], "mapped_scaling_cos": [],
            "r1": [], "r2": [], "r3": [], "scalar_reward": [],
            "cos_delta": [], "cos_b_b": [],
            "l2_delta": [], "l2_b_b": [],
            "model_cos_delta": [], "model_cos_b_b": [],
        }

        for key, default in keys_to_init.items():
            if key not in self.full_history:
                self.full_history[key] = default

        # Append values
        self.full_history["s_b"].append(s_b)
        self.full_history["s_hat"].append(s_hat)
        self.full_history["benign_norm"].append(benign_norm)
        self.full_history["server_update_norm"].append(server_update_norm)
        self.full_history["posterior"].append(posterior)
        self.full_history["scaling_factor"].append(self.scaling_l2)
        self.full_history["epsilon"].append(self.epsilon)
        self.full_history["norm_scaling_l2"].append(norm_scaling_l2)
        self.full_history["norm_scaling_cos"].append(norm_scaling_cos)
        self.full_history["mapped_scaling_l2"].append(mapped_scaling_l2)
        self.full_history["mapped_scaling_cos"].append(mapped_scaling_cos)

        self.full_history["r1"].append(r1)
        self.full_history["r2"].append(r2)
        self.full_history["r3"].append(r3)
        self.full_history["scalar_reward"].append(scalar_reward)
        self.full_history["cos_delta"].append(cos_delta)
        self.full_history["cos_b_b"].append(cos_b_b)
        self.full_history["l2_delta"].append(l2_delta)
        self.full_history["l2_b_b"].append(l2_b_b)
        self.full_history["model_cos_delta"].append(model_cos_delta)
        self.full_history["model_cos_b_b"].append(model_cos_b_b)


    def log_attack_statistics(self, logger, benign_update, malicious_update, avg_l2_dist, avg_cos_distance,
                                avg_full_model_cos, server_model): 
        mali_norm = torch.norm(malicious_update).item()
        benign_norm = torch.norm(benign_update).item()
        l2_dist = torch.norm(malicious_update - benign_update).item()
        cos_dist = 1 - torch.nn.functional.cosine_similarity(
            malicious_update.unsqueeze(0), benign_update.unsqueeze(0)
        ).item()

        server_weights = flatten_model(server_model.state_dict())
        mali_model_vec = server_weights + malicious_update
        benign_model_vec = server_weights + benign_update

        cos_model_dist = 1 - torch.nn.functional.cosine_similarity(
            mali_model_vec.unsqueeze(0), benign_model_vec.unsqueeze(0)
        ).item()

        logger.info(
            f"[FedOda] Norms     |  norm(Δ_m): {mali_norm:.4f} | norm(Δ_b): {benign_norm:.4f}"
        )
        logger.info(
            f"[FedOda] L2 Dist   | L2(Δ_m, Δ_b): {l2_dist:.4f} | L2(Δ_b_i, Δ_b_j): {avg_l2_dist:.4f}"
        )
        logger.info(
            f"[FedOda] Cos Dist Δ|  Δ_m~Δ_b: {cos_dist:.4f} | Δb~Δb: {avg_cos_distance:.4f}"
        )
        logger.info(
            f"[FedOda] Cos Dist M| model_m~model_b: {cos_model_dist:.8f} | model_b~model_b: {avg_full_model_cos:.8f}"
        )


    def _regular_attack(self, sample_num, batch_size, mco_dict, c_epochs):
        # Step1. perform analysis
        self.logger.info("[FedOda] Step 1: Computing best attack direction (g_m)")
        g_m_dict, g_m = self.get_attack_direction(model=self.server_model,
                                                  train_loader=self.pooled_loader,
                                                  initial_lr=0.001,
                                                  num_epochs=2)

        self.logger.info("[FedOda] Step 2: Predicting benign update direction with pooled data")
        benign_direction, simulated_norm = self.simulate_benign_direction(
            loader=self.pooled_loader,
            data_size=self.client_data_size,
            batch_size=self.client_batch_size
        )

        self.logger.info("[FedOda] Step 3: Simulating benign updates for sampled clients")
        avg_l2_distance, avg_cos_distance, avg_l2_norm, avg_full_model_cos = \
            self.simulate_benign_updates_l2_cos(sample_num, batch_size, mco_dict, c_epochs)

        # Get the estimated benign model update(\Delta_b'), and the best attack direction (g_m)   
        benign_update = avg_l2_norm * benign_direction
        mali_direction = g_m
        if self.prev_server_model!=None:
            server_update_norm = self._calculate_update_distance(
                    self.prev_server_model.state_dict(), 
                    self.server_model.state_dict(), 
                    distance_type='l2'
                )
        else:
            server_update_norm = 0.0
        
        round_number = len(self.val_acc_pooled_history)
        # print(round_number, self.val_acc_pooled_history)
        acc_curr = self.val_acc_pooled_history[-1] if round_number > 1 else 0
        acc_prev = self.val_acc_pooled_history[-2] if round_number > 1 else acc_curr
        malicious_participation_rate = self.current_s_m / self.malicious_clients_num      

        if self.attack_mode == "BO":
            malicious_update_dict, malicious_update, reward_vector, action = bo_attack(
                fedoda=self,
                mali_direction=mali_direction,
                benign_update=benign_update,
                avg_cos_distance=avg_cos_distance,
                avg_l2_distance=avg_l2_distance,
                avg_update_norm=avg_l2_norm,
                prev_acc=acc_prev,
                curr_acc=acc_curr
            )
            r1, r2, r3 = reward_vector.tolist()
            reward_scalar = r2
        elif self.attack_mode == "BI":  # Default BI mode
            self.epsilon = self.scaling_l2 * avg_l2_norm
            malicious_update_dict, malicious_update = bi_attack(
                self, mali_direction, benign_update,
                avg_cos_distance, avg_l2_norm
            )
        elif self.attack_mode == "PCAC":
            malicious_update_dict, malicious_update, action, state, noise_std = pcac_attack(
                fedoda=self,
                mali_direction=mali_direction,
                benign_update=benign_update,
                avg_cos_distance=avg_cos_distance,
                avg_l2_distance=avg_l2_distance,
                avg_update_norm=avg_l2_norm,
                prev_acc=acc_prev,
                curr_acc=acc_curr
            )

            self.logger.info(
                f"[FedOda] PCAC noise_std: {noise_std:.4f}, "
                f"action: {[f'{float(x):.4f}' for x in action]}, "
                f"state: {[f'{float(x):.4f}' for x in state]}"
            )

            if self.prev_acc is not None:
                r1 = 1.0 - acc_curr
                r2 = (self.prev_acc - acc_curr) / 0.2
                r3 = 0
                reward_vector = [r1, r2]
                reward_scalar = np.dot(reward_vector, self.pcac_preference_vector.numpy())

                self.pcac_agent.store_transition(
                    s=self.prev_state,
                    a=self.prev_action,
                    r=reward_vector,
                    s_next=state,
                    preference_vector=self.pcac_preference_vector
                )
                self.pcac_agent.update()
            else:
                # First iteration: can't compute rewards that depend on prev_acc
                r1 = 0.0
                r2 = 0.0
                r3 = 0.0
                reward_scalar = 0.0
            self.prev_state = state
            self.prev_action = action
            self.prev_acc = acc_curr

            
        elif self.attack_mode == "MOQ":
            malicious_update_dict, malicious_update, action, state = moq_attack(
                fedoda=self,
                mali_direction=mali_direction,
                benign_update=benign_update,
                avg_cos_distance=avg_cos_distance,
                avg_l2_distance=avg_l2_distance,
                avg_update_norm=avg_l2_norm,
                prev_acc=acc_prev,
                curr_acc=acc_curr
            )
            self.logger.info(f"[FedOda] MOQ action: {[float(x) for x in action]}, state: {state}")

            # === Step 5: Compute reward from previous round
            if self.prev_acc is not None:
                r1 = 1.0 - acc_curr
                # delta_r_threshold = 0.05
                # if (self.prev_acc - acc_curr) >= delta_r_threshold:
                #     r2 = (self.prev_acc - acc_curr) * 5
                # else:
                #     r2 = 0  # or whatever default value you want when the condition isn't met
                r2 = (self.prev_acc - acc_curr) * 5
                r3 = 0
                reward_vector = [r1, r2]
                
                self.logger.info(f"[FedOda] Reward r1: {r1:.4f}, r2: {r2:.4f} (prev_acc={self.prev_acc:.4f}, acc_curr={acc_curr:.4f})")
                old_eps, new_eps = self.moq_agent.adjust_epsilon(
                    r1=r1,
                    r2=r2,
                    r1_threshold=0.5,
                    r2_threshold=0.15,
                    epsilon_cap=0.5
                )
                self.logger.info(f"[FedOda] Adjusted epsilon: {old_eps:.4f} → {new_eps:.4f}")
                
                scalar_reward = np.dot(reward_vector, self.moq_preference_vector.numpy())
                self.logger.info(f"[FedOda] Scalar reward: {scalar_reward:.4f}")
                
                self.moq_agent.update(
                    s=self.prev_state,
                    a=self.prev_action,
                    r=reward_vector,
                    s_next=state,
                    preference_vector= self.moq_preference_vector
                )
            else:
                r1 = 0
                r2 = 0
                r3 = 0
                reward_vector = [r1, r2]
            reward_scalar = r1 + r2 

            # === Step 6: Save for next round
            self.prev_state = state
            self.prev_action = action
            self.prev_acc = acc_curr

            round_number = len(self.val_acc_pooled_history)

            if round_number % 5 == 0:
                with torch.no_grad():
                    device = self.moq_agent.device
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    preference_vector = self.moq_preference_vector.to(device)

                    action_tensor = torch.tensor(self.moq_agent.action_space, dtype=torch.float32).to(device)
                    state_rep = state_tensor.repeat(len(self.moq_agent.action_space), 1)

                    q_values_vec = self.moq_agent.q_net(state_rep, action_tensor)  # [N, reward_dim]
                    scalar_q_values = torch.matmul(q_values_vec, preference_vector)  # [N]

                    best_action_idx = torch.argmax(scalar_q_values).item()
                    best_action = self.moq_agent.action_space[best_action_idx]

                    self.logger.info(f"[FedOda][Policy Check] Round {round_number}: Best Action (Q*) = {best_action}")

        elif self.attack_mode == "PPO":
            # === Step 0: Measure current state
            curr_state = torch.tensor([
                acc_prev,
                avg_cos_distance / 2.0,
                avg_l2_norm / 100.0,
                round_number / 500.0,
                malicious_participation_rate
            ], dtype=torch.float32)

            # === Step 1: Select action for current round
            action, log_prob = self.ppo_agent.select_action(curr_state)

            # Save current state/action/log_prob for next round
            self.ppo_prev_state = curr_state
            self.ppo_prev_action = action
            self.ppo_prev_log_prob = log_prob

            # === Step 2: Perform attack using selected action → get reward from here
            malicious_update_dict, malicious_update, reward_vector = guided_attack_from_action(
                fedoda=self,
                round_number=round_number,
                prev_acc=acc_prev,
                curr_acc=acc_curr,
                avg_cos_distance=avg_cos_distance,
                avg_l2_distance=avg_l2_distance,
                avg_update_norm=avg_l2_norm,
                avg_full_model_cos=avg_full_model_cos,
                malicious_participation_rate=malicious_participation_rate,
                mali_direction=mali_direction,
                benign_update=benign_update,
                action=action,
                log_prefix="[PPO]"
            )
            
            # ✅ Always unpack reward_vector
            r1, r2, r3 = reward_vector.tolist()

            # === Step 3: Log previous transition with reward from current attack
            if hasattr(self, "ppo_prev_state"):
                reward_scalar = self.ppo_agent.combine_reward(round_number - 1, reward_vector)

                self.logger.info(
                    f"[FedOda] PPO reward for previous round {round_number - 1}: {reward_scalar:.4f}"
                )

                self.ppo_agent.buffer.append((
                    self.ppo_prev_state.detach(),
                    self.ppo_prev_action.detach(),
                    reward_vector.detach(),
                    self.ppo_prev_log_prob.detach()
                ))
            else:
                self.logger.warning("[FedOda] PPO previous state/action/log_prob not set. Skipping logging.")
                reward_scalar = 0 

            # === Step 4: Train PPO every few rounds
            if round_number >= 5 and round_number % 2 == 0:
                self.logger.info(f"[FedOda][PPO] Training PPO agent at round {round_number}")
                self.ppo_agent.train(memory=self.ppo_agent.buffer, round_number=round_number)
                self.ppo_agent.buffer.clear()
        elif self.attack_mode == "ES":
            round_number = len(self.val_acc_pooled_history)
            acc_curr = self.val_acc_pooled_history[-1] if round_number > 1 else 0
            acc_prev = self.val_acc_pooled_history[-2] if round_number > 1 else acc_curr

            # === Step 0: Initialize ES agent if not yet
            if not hasattr(self, "es_agent"):
                from es_agent import ESAgent
                self.es_agent = ESAgent(
                    init_action=np.array([0.0, -1.0]),
                    sigma=0.1,
                    alpha=0.05,
                    bounds=[(-1.0, 1.0), (-1.0, 1.0)]  # per-dimension bounds
                )
                self.logger.info("[FedOda][ES] Initialized Evolution Strategy Agent.")

            # === Step 1: Sample ONE action
            action = self.es_agent.sample_action()

            # === Step 2: Run attack
            malicious_update_dict, malicious_update, reward_vector = guided_attack_from_action(
                fedoda=self,
                round_number=round_number,
                prev_acc=acc_prev,
                curr_acc=acc_curr,
                avg_cos_distance=avg_cos_distance,
                avg_l2_distance=avg_l2_distance,
                avg_update_norm=avg_l2_norm,
                avg_full_model_cos=avg_full_model_cos,
                malicious_participation_rate=self.current_s_m / self.malicious_clients_num,
                mali_direction=mali_direction,
                benign_update=benign_update,
                action=torch.tensor(action, dtype=torch.float32),
                log_prefix="[ES]"
            )

            # ✅ Always unpack reward_vector
            r1, r2, r3 = reward_vector.tolist()
            reward_scalar = float(r1 + r2 + r3)

        # Shared logging
        self.log_attack_statistics(
            logger=self.logger,
            benign_update=benign_update,
            malicious_update=malicious_update,
            avg_l2_dist=avg_l2_distance,
            avg_cos_distance=avg_cos_distance,
            avg_full_model_cos=avg_full_model_cos,
            server_model=self.server_model,
        )
        
        self.log_to_full_history(
            s_b=None,
            s_hat=None,
            benign_norm=avg_l2_norm,
            server_update_norm=server_update_norm,
            r1=r1,
            r2=r2,
            r3=r3,
            scalar_reward=reward_scalar,
            cos_delta=None,
            cos_b_b=avg_cos_distance,
            l2_delta=None,
            l2_b_b=avg_l2_distance,
            model_cos_delta=None,
            model_cos_b_b=avg_full_model_cos,
            posterior=None,
            norm_scaling_l2=float(action[0]),
            norm_scaling_cos=float(action[1]),
            mapped_scaling_l2=self.scaling_l2,
            mapped_scaling_cos=self.scaling_cos,
        )

        self.prev_s_m = self.current_s_m
        
        # Store for next round
        self.prev_server_model = deepcopy(self.server_model)
        self.prev_benign_update = benign_update.clone().detach()
        self.prev_malicious_update = malicious_update.clone().detach()

        return malicious_update_dict, malicious_update