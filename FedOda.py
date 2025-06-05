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
from skopt import Optimizer
from skopt.space import Real
from utilities import create_balanced_loader


class FedOda:
    def __init__(self, model_fn, device, pooled_loader, vali_loader, total_clients_num, 
                 malicious_client_ids, fl_mali_data_dict, E=5, scaling_factor=8,
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
        self.attack_mode = "BO" # 'BO', "BI"
        
        # Create the Bayesian Optimizer
        if self.attack_mode == "BO":
            l2_bound = (0.0, 1.0)
            cos_bound = (0.0, 1.0)
            self.bayes_optimizer = BayesOptimizer(
                bounds=[l2_bound, cos_bound],
                base_estimator="GP",
                acq_func="EI",
                random_state=42
            )
            
        # For tracking performance history and reward scores
        self.bo_reward_history = []  # stores rewards (with None if skipped)
        self.bo_param_history = []   # stores chosen params per round
        
        # State variables
        self.model0_sd = None
        self.modelt_sd = None
        self.att_direction = None
        self.last_update_norm = 0
        self.benign_direction = None
        self.scaling_factor = scaling_factor
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


    def get_attack_direction(self, model, train_loader, max_retries=50, initial_lr=0.001):
        def contains_nan(tensor_list):
            return any(param is not None and torch.isnan(param).any() for param in tensor_list)

        model = deepcopy(model).to(self.device)
        model.train()
        loss_fn = lambda outputs, labels: -F.cross_entropy(outputs, labels)

        # Move model state to CPU to reduce GPU memory usage
        original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        retry_count = 0
        current_lr = initial_lr
        optimizer = optim.SGD(model.parameters(), lr=current_lr)

        while retry_count < max_retries:
            retry_count += 1
            # Restore model state to GPU
            model.load_state_dict({k: v.to(self.device) for k, v in original_state.items()})
            nan_detected = False

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
                current_lr *= 0.1
                optimizer = optim.SGD(model.parameters(), lr=current_lr)
                print(f"[Retry {retry_count}] NaN detected. Reducing lr to {current_lr} and restarting.")
            else:
                break

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

        # Restore original model state to GPU
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



    def adjust_epsilon(self, epsilon, posterior):
        if posterior < self.threshold:
            return epsilon * self.reduce_factor
        else:
            return epsilon * self.increase_factor


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
        temp_loader = DataLoader(list(zip(inputs, labels)), batch_size=batch_size, shuffle=True)

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
                                shuffle=True, num_workers=5, drop_last=True)
            
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
        return 100 * correct / total



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
            return None, None, None

        if self.previous_server_model is None:
            return self._first_iteration_attack()
        else:
            # ["classifier.weight"]
            return self._regular_attack(sample_num, batch_size, mco_dict, c_epochs, target_layers=None)


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


    def log_attack_statistics(self, logger, benign_update, malicious_update, avg_l2_dist, avg_cos_distance, avg_full_model_cos, server_model):
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
            f"[FedOda] Norms     | norm(Δ_b): {benign_norm:.4f} | norm(Δ_m): {mali_norm:.4f} "
        )
        logger.info(
            f"[FedOda] L2 Dist   | L2(Δ_m, Δ_b): {l2_dist:.4f} | L2(Δ_b_i, Δ_b_j): {avg_l2_dist:.4f}"
        )
        logger.info(
            f"[FedOda] Cos Dist Δ| Δb~Δb: {avg_cos_distance:.4f} | Δ_m~Δ_b: {cos_dist:.4f} "
        )
        logger.info(
            f"[FedOda] Cos Dist M| model_m~model_b: {cos_model_dist:.8f} | model_b~model_b: {avg_full_model_cos:.8f}"
        )


    def _first_iteration_attack(self):
        self.logger.info("[FedOda] First iteration detected. Performing special attack initialization.")

        # [Previous steps 1-3 remain the same...]
        g_m_dict, g_m_vec = self.get_attack_direction(self.server_model, self.pooled_loader)
        benign_dir, benign_norm = self.simulate_benign_direction(
            loader=self.pooled_loader,
            data_size=self.client_data_size,
            batch_size=self.client_batch_size
        )
        self.epsilon = self.scaling_factor * benign_norm
        malicious_update = self.epsilon * g_m_vec

        # Evaluate the malicious update
        val_accuracy = self._evaluate_malicious_update(malicious_update)
        self.logger.info(f"[FedOda] Malicious update validation accuracy: {val_accuracy:.2f}%")
        
        malicious_update_dict = vector_to_state_dict(malicious_update, self.server_model)
        self.logger.info(f"[FedOda] First-round attack completed. Returning malicious update.")
        posterior = None
        # self.update_history(None, None, None, None, malicious_update)
        self.prev_s_m = self.current_s_m
        return malicious_update_dict, malicious_update, posterior


    def _regular_attack(self, sample_num, batch_size, mco_dict, c_epochs, target_layers="none"):
        self.logger.info("[FedOda] Step 1: Computing attack direction (g_m)")
        g_m_dict, g_m_vec = self.get_attack_direction(self.server_model, self.pooled_loader)

        self.logger.info("[FedOda] Step 2: Predicting benign update")
        benign_dir, simulated_norm = self.simulate_benign_direction(
            loader=self.pooled_loader,
            data_size=self.client_data_size,
            batch_size=self.client_batch_size
        )

        avg_l2_distance, avg_cos_distance, avg_update_norm, avg_full_model_cos = \
            self.simulate_benign_updates_l2_cos(sample_num, batch_size, mco_dict, c_epochs)
        benign_update = avg_update_norm * benign_dir
        avg_l2_norm = avg_update_norm
        mali_direction = g_m_vec

        val_accuracy = self.validate(self.server_model, self.pooled_balanced_loader)

        if self.attack_mode == "BO":
            # 1. Compute cosine distance between directions (benign_dir and g_m_vec)
            cos_sim_dirs = F.cosine_similarity(
                benign_dir.unsqueeze(0),
                g_m_vec.unsqueeze(0),
                dim=1
            ).item()
            cos_dist_dirs = 1 - cos_sim_dirs

            # 2. Get normalized suggestions in [0, 1]
            x_next = self.bayes_optimizer.suggest_next_point()
            norm_scaling_l2, norm_scaling_cos = x_next[0], x_next[1]

            # 3. Map normalized values to actual bounds (LOG-scaled)
            self.scaling_l2 = log2_map(norm_scaling_l2, 0.5, 10.0)
            cos_dist_low_bound = avg_cos_distance*0.9
            self.logger.info(f"[FedOda][BO] Mapped cos domain: [{cos_dist_low_bound:.4f}, {cos_dist_dirs:.4f}]")
            self.scaling_cos = log2_map(norm_scaling_cos, cos_dist_low_bound, cos_dist_dirs)

            # Append the new metrics to history
            self.full_history["norm_scaling_l2"].append(norm_scaling_l2)
            self.full_history["norm_scaling_cos"].append(norm_scaling_cos)
            self.full_history["mapped_scaling_l2"].append(self.scaling_l2)
            self.full_history["mapped_scaling_cos"].append(self.scaling_cos)

            # 4. Assign and apply
            self.epsilon = self.scaling_l2 * avg_l2_norm

            self.logger.info(
                f"[FedOda][BO] norm_scaling_l2={norm_scaling_l2:.4f}, norm_scaling_cos={norm_scaling_cos:.4f}"
            )
            self.logger.info(
                f"[FedOda][BO] mapped_scaling_l2={self.scaling_l2:.4f}, mapped_scaling_cos={self.scaling_cos:.4f}, cos_dirs={cos_dist_dirs:.4f}"
            )

            # 5. Use scaled cosine budget safely
            malicious_update = self.craft_malicious_update_cos_budget(
                mali_direction, benign_update, self.scaling_cos
            )
            malicious_update = malicious_update / (malicious_update.norm() + 1e-10) * self.epsilon
            
            if not hasattr(self, "val_acc_history"):
                self.val_acc_history = []
            self.val_acc_history.append(val_accuracy)

            reward = calculate_bo_reward(
                acc_history=self.val_acc_history,
                current_round=len(self.val_acc_history)-1,
                gamma=0.98,
                warmup_rounds=3,
                logger=self.logger
            )
            
            if reward is not None:
                self.bayes_optimizer.observe(x_next, reward)
                self.logger.info(f"[FedOda][BO] Observed reward: {reward:.4f} for x={x_next}")
            else:
                self.logger.info(f"[FedOda][BO] Warmup round — skipping reward observation for x={x_next}")
        
        elif self.attack_mode == "BI":  # Default BI mode
            self.epsilon = self.scaling_factor * avg_l2_norm
            malicious_update = self.craft_malicious_update_cos_budget(
                mali_direction, benign_update, avg_cos_distance
            )
            malicious_update = malicious_update / (malicious_update.norm() + 1e-10) * self.epsilon

            # For BI mode, we might not have all these values, so we can append None or some default
            self.full_history["norm_scaling_l2"].append(None)
            self.full_history["norm_scaling_cos"].append(None)
            self.full_history["mapped_scaling_l2"].append(None)
            self.full_history["mapped_scaling_cos"].append(None)

        val_accuracy = self._evaluate_malicious_update(malicious_update)
        self.logger.info(f"[FedOda] Malicious update validation accuracy: {val_accuracy:.4f}")

        # Assemble malicious update dict
        malicious_update_dict = vector_to_state_dict(malicious_update, self.server_model)

        # Shared logging
        self.log_attack_statistics(
            logger=self.logger,
            benign_update=benign_update,
            malicious_update=malicious_update,
            avg_l2_dist=avg_l2_distance,
            avg_cos_distance=avg_cos_distance,
            avg_full_model_cos=avg_full_model_cos,
            server_model=self.server_model
        )

        self.prev_s_m = self.current_s_m
        posterior = None if self.attack_mode == "BO" else 1  # Optional: handle posterior from BI inference if needed
        return malicious_update_dict, malicious_update, posterior
    


    def _regular_attack_old(self, sample_num, batch_size, mco_dict, c_epochs, target_layers="none"):
        self.logger.info("[FedOda] Step 1: Computing attack direction (g_m)")
        g_m_dict, g_m_vec = self.get_attack_direction(self.server_model, self.pooled_loader)

        self.logger.info("[FedOda] Step 2: Predicting benign update")
        benign_dir, simulated_norm = self.simulate_benign_direction(
            loader=self.pooled_loader,
            data_size=self.client_data_size,
            batch_size=self.client_batch_size
        )
        
        avg_l2_distance, avg_cos_distance, avg_update_norm, avg_full_model_cos = \
            self.simulate_benign_updates_l2_cos(sample_num, batch_size, mco_dict, c_epochs)
        benign_update = avg_update_norm * benign_dir

        # For both BI and BO: use avg_l2_norm from above
        avg_l2_norm = avg_update_norm       

        s_b, s_hat, s_m, alpha = self.estimate_client_distribution()
        
        # Step 5: Compute ε = scaling_factor * benign_norm
        self.epsilon = self.scaling_factor * avg_l2_norm
        mali_direction = g_m_vec

        # Clip the malicious update to ensure it stays within the cosine budget
        malicious_update = self.craft_malicious_update_cos_budget(mali_direction, benign_update, avg_cos_distance)
        self.logger.info(f"[FedOda] Crafted malicious update with cosine budget: {avg_cos_distance:.4f}")
        malicious_update = malicious_update / (malicious_update.norm() + 1e-10) * self.epsilon

        # Evaluate the malicious update
        val_accuracy = self._evaluate_malicious_update(malicious_update)
        self.logger.info(f"[FedOda] Malicious update validation accuracy: {val_accuracy:.2f}%")
        self.logger.info(f"[FedOda] benign_norm = {simulated_norm:.4f}, scaling_factor = {self.scaling_factor:.4f}, ε = {self.epsilon:.4f}")

        # Step 7: If selected last round, update history and perform inference
        if self.prev_s_m is not None and self.prev_s_m > 0:
            self.logger.info("[FedOda] Step 7: Observing actual server update from previous round")
            delta_s = flatten_model(self.server_model.state_dict()) - flatten_model(self.previous_server_model.state_dict())
            delta_s_pred = self.estimate_server_update(s_b, s_m, benign_update, malicious_update)

            self.logger.info("[FedOda] Step 8: Updating history")
            self.update_history(delta_s, benign_update, delta_s_pred, s_m, malicious_update)

            cos_d1, cos_d2, cos_d3, cos_d4 = self.get_cos_dist(delta_s,
                                                            benign_update,
                                                            delta_s_pred,
                                                            malicious_update)

            l2_d1, l2_d2, l2_d3, l2_d4 = self.get_l2_dist(delta_s,
                                                        benign_update,
                                                        delta_s_pred,
                                                        malicious_update)

            self.logger.info(
                f"[FedOda] cos_d(Δs, Δb')={cos_d1:.4f}, "
                f"cos_d(Δs, Δs')={cos_d2:.4f}, "
                f"cos_d(Δs, Δm)={cos_d3:.4f}, "
                f"cos_d(Δb', Δm)={cos_d4:.4f}"
            )

            self.logger.info(
                f"[FedOda] L2²(Δs, Δb')={l2_d1:.4f}, "
                f"L2²(Δs, Δs')={l2_d2:.4f}, "
                f"L2²(Δs, Δm)={l2_d3:.4f}, "
                f"L2²(Δb', Δm)={l2_d4:.4f}"
            )

            self.logger.info("[FedOda] Step 9: Performing Bayesian inference")
            if self.inference_mode == 'fixed':
                posterior = None
                self.logger.info("[FedOda] Fixed mode - maintaining constant scaling factor")
            elif self.inference_mode == 'cos':
                posterior, mean_log_likelihood = self.bayesian_inference_cosine()
            elif self.inference_mode == 'l2':
                posterior, mean_log_likelihood = self.bayesian_inference_l2(d=len(malicious_update), use_average=False)
            elif self.inference_mode == 'sign':
                posterior = self.sign_based_bayesian_inference()
            elif self.inference_mode == 'validation':
                # Validate current server model on pooled data
                current_val_acc = self.validate(self.server_model, self.pooled_loader)
                self.pooled_val.append(current_val_acc)
                posterior = 1
                self.logger.info(f"[FedOda] Validation-based inference mode - current pooled val acc: {current_val_acc:.2f}%")

                if len(self.pooled_val) >= 2:
                    prev_acc = self.pooled_val[-2]
                    delta_acc = current_val_acc - prev_acc

                    self.logger.info(f"[FedOda] Previous Val Acc: {prev_acc:.2f}%, Delta: {delta_acc:.2f}%")

                    if current_val_acc <= 10.0:
                        self.logger.info("[FedOda] Accuracy very low, attack already effective. Keeping scaling factor unchanged.")
                        posterior = 1
                    elif delta_acc < 0:
                        self.logger.info("[FedOda] Accuracy dropped - attack effective. Slight adjustment to scaling factor.")
                        # self.scaling_factor *= 1.1  # Keep or slightly fine-tune (e.g., *= 0.99 if needed)
                        posterior = 1
                    else:
                        self.logger.info("[FedOda] Accuracy did not drop - attack ineffective. Increasing scaling factor.")
                        # self.scaling_factor *= 0.8  # Increase scaling to make future attack stronger
                        posterior = 0
                else:
                    self.logger.info("[FedOda] Not enough validation history - skipping scaling factor adjustment.")
            else:
                raise ValueError(f"Unknown inference mode: {self.inference_mode}")

            # Store posterior in history
            if not hasattr(self, "posterior_history"):
                self.posterior_history = []
            self.posterior_history.append(posterior)
            if len(self.posterior_history) > self.E:
                self.posterior_history.pop(0)

            self.log_to_full_history(
                s_b=None,
                s_hat=None,
                benign_norm=simulated_norm,
                server_update_norm=delta_s.norm().item(),
                cos_d1=cos_d1, cos_d2=cos_d2, cos_d3=cos_d3, cos_d4=cos_d4,
                l2_d1=l2_d1, l2_d2=l2_d2, l2_d3=l2_d3, l2_d4=l2_d4,
                l=mean_log_likelihood if self.inference_mode == "l2" else None,
                posterior=posterior
            )

            if self.inference_mode in ['l2', 'cos', 'sign', 'validation']:
                valid_posteriors = [p for p in self.posterior_history if p is not None]
                if valid_posteriors:
                    prev_factor = self.scaling_factor
                    self.scaling_factor = self.adjust_scaling_factor(prev_factor,
                                                                     exploration_prob=0.1,
                                                                     exploration_scale=1.5)
                    self.logger.info(f"[FedOda] Scaling factor adjusted: {prev_factor:.4f} -> {self.scaling_factor:.4f}")
                else:
                    self.logger.info("[FedOda] No valid posteriors available — skipping scaling factor adjustment.")
            elif self.inference_mode == 'fixed':
                self.logger.info(f"[FedOda] Fixed mode - scaling factor remains at {self.scaling_factor:.4f}")
        else:
            self.logger.info("[FedOda] Malicious clients were not selected last round. Skipping inference.")
            posterior = None
            if not hasattr(self, "posterior_history"):
                self.posterior_history = []
            self.posterior_history.append(None)
            if len(self.posterior_history) > self.E * 2:
                self.posterior_history.pop(0)

        # Step 10: Return malicious update
        self.logger.info("[FedOda] Step 10: Creating malicious update dict")
        malicious_update_dict = vector_to_state_dict(malicious_update, self.server_model)
        
        if target_layers != None:
            # Convert benign update to dict
            benign_update_dict = vector_to_state_dict(benign_update, self.server_model)

            for k in malicious_update_dict:
                if k not in target_layers:
                    # self.logger.info(f"[FedOda] Layer {k} not in target layers, skipping.")
                    malicious_update_dict[k] = benign_update_dict[k]  # Use benign layer instead
                else:
                    self.logger.info(f"[FedOda] Layer {k} in target layers, keeping malicious update.")
            
            # Reflatten the updated malicious dict
            malicious_update = flatten_model(malicious_update_dict)
        

        # 1. Norms
        mali_norm = torch.norm(malicious_update).item()
        benign_norm = torch.norm(benign_update).item()

        # 2. L2 distance
        l2_dist = torch.norm(malicious_update - benign_update).item()

        # 3. Cosine distance between updates
        cos_dist = 1 - torch.nn.functional.cosine_similarity(malicious_update.unsqueeze(0), benign_update.unsqueeze(0)).item()

        # 4. Cosine distance between (update + model)
        server_weights = flatten_model(self.server_model.state_dict())
        mali_model_vec = server_weights + malicious_update
        benign_model_vec = server_weights + benign_update

        cos_model_dist = 1 - torch.nn.functional.cosine_similarity(mali_model_vec.unsqueeze(0), benign_model_vec.unsqueeze(0)).item()
        
        self.logger.info(
            f"[FedOda] Norms     | norm(Δ_b): {benign_norm:.4f} | norm(Δ_m): {mali_norm:.4f} "
        )
        self.logger.info(
            f"[FedOda] L2 Dist   | L2(Δ_m, Δ_b): {l2_dist:.4f} | L2(Δ_b_i, Δ_b_j): {avg_l2_norm:.4f}"
        )
        self.logger.info(
            f"[FedOda] Cos Dist Δ| Δb~Δb: {avg_cos_distance:.4f} | Δ_m~Δ_b: {cos_dist:.4f} "
        )
        self.logger.info(
            f"[FedOda] Cos Dist M| model_m~model_b: {cos_model_dist:.8f} | model_b~model_b: {avg_full_model_cos:.8f}"
        )

        self.logger.info(f"[FedOda] Posterior P(H1|D) = {posterior}, new scaling_factor = {self.scaling_factor:.4f}, ε recomputed = {self.scaling_factor * avg_l2_norm:.4f}")
        self.logger.info("[FedOda] Attack completed, returning malicious update.")

        self.prev_s_m = self.current_s_m

        return malicious_update_dict, malicious_update, posterior


    def log_to_full_history(self, *, s_b=None, s_hat=None,
                            benign_norm=None, server_update_norm=None,
                            cos_d1=None, cos_d2=None, cos_d3=None, cos_d4=None,
                            l2_d1=None, l2_d2=None, l2_d3=None, l2_d4=None,
                            l=None, posterior=None):
        """
        Logs a single iteration's key metrics to self.full_history.
        """

        if not hasattr(self, "full_history"):
            self.full_history = {
                "s_b": [], "s_hat": [],
                "benign_norm": [], "server_update_norm": [],
                "cos_d1": [], "cos_d2": [], "cos_d3": [], "cos_d4": [],
                "l2_d1": [], "l2_d2": [], "l2_d3": [], "l2_d4": [],
                "mean_log_likelihood": [], "posterior": [],
                "scaling_factor": [], "epsilon": []
            }

        self.full_history["s_b"].append(s_b)
        self.full_history["s_hat"].append(s_hat)
        self.full_history["benign_norm"].append(benign_norm)
        self.full_history["server_update_norm"].append(server_update_norm)
        self.full_history["cos_d1"].append(cos_d1)
        self.full_history["cos_d2"].append(cos_d2)
        self.full_history["cos_d3"].append(cos_d3)
        self.full_history["cos_d4"].append(cos_d4)
        self.full_history["l2_d1"].append(l2_d1)
        self.full_history["l2_d2"].append(l2_d2)
        self.full_history["l2_d3"].append(l2_d3)
        self.full_history["l2_d4"].append(l2_d4)        
        self.full_history["mean_log_likelihood"].append(l)
        self.full_history["posterior"].append(posterior)
        self.full_history["scaling_factor"].append(self.scaling_factor)
        self.full_history["epsilon"].append(self.epsilon)


# Utility
def flatten_model(state_dict):
    """Flatten all tensors in a state_dict (including buffers)."""
    return torch.cat([
        param.view(-1).float() for param in state_dict.values()
        if isinstance(param, torch.Tensor)
    ])

def vector_to_state_dict(vector, reference_model):
    """
    Convert a flattened vector into a full state_dict (including buffers)
    that matches the structure of reference_model.state_dict().
    """
    ref_sd = reference_model.state_dict()
    new_sd = {}
    idx = 0
    for key, param in ref_sd.items():
        if isinstance(param, torch.Tensor):
            numel = param.numel()
            new_sd[key] = vector[idx: idx + numel].view_as(param).clone().detach()
            idx += numel
        else:
            # Non-tensor (rare), copy as-is
            new_sd[key] = param
    return new_sd


def calculate_bo_reward(
    acc_history: List[float],
    current_round: int,
    gamma: float = 0.95,
    warmup_rounds: int = 3,
    logger=None
) -> Optional[float]:
    """
    Time-discounted reward function for BO-based attack effectiveness.
    Uses exponential discounting based only on current round (no need for total_rounds).
    Incorporates current accuracy to reward drops more when accuracy is already low.

    Args:
        acc_history (List[float]): Accuracy history across rounds.
        current_round (int): Current FL round (0-based).
        gamma (float): Time discount factor (0 < gamma <= 1).
        warmup_rounds (int): Number of rounds to skip before reward calculation.
        logger: Optional logger for debugging.

    Returns:
        Optional[float]: The calculated reward, or None if warmup not completed.
    """
    if logger:
        logger.info(f"Calculating BO reward at round {current_round}")
        logger.info(f"Accuracy history: {acc_history}")

    if len(acc_history) < 2 or current_round < warmup_rounds:
        if logger:
            logger.info("Skipping reward calculation (warmup or insufficient history)")
        return None

    current_acc = acc_history[-1]
    prev_acc = acc_history[-2]
    delta_acc = prev_acc - current_acc

    # Time-based discount (no total_rounds needed)
    discount = gamma ** current_round

    # Accuracy-based scaling (lower accuracy → higher reward scaling)
    acc_scale = 1.0 - (current_acc / 100.0)
    acc_scale = max(0.0, min(acc_scale, 1.0))  # Clamp to [0, 1]

    reward = delta_acc * acc_scale * discount

    if logger:
        logger.info(f"Prev acc: {prev_acc:.4f}, Curr acc: {current_acc:.4f}, Δacc: {delta_acc:.4f}")
        logger.info(f"Acc scale: {acc_scale:.4f}, Discount: {discount:.6f}")
        logger.info(f"Final reward: {reward:.6f}")

    return reward



class BayesOptimizer:
    def __init__(self, bounds, base_estimator="GP", acq_func="EI", random_state=None):
        self.optimizer = Optimizer(
            dimensions=bounds,
            base_estimator=base_estimator,
            acq_func=acq_func,
            random_state=random_state
        )
        self.xs = []
        self.ys = []

    def suggest_next_point(self):
        x_next = self.optimizer.ask()
        self.xs.append(x_next)
        return x_next

    def observe(self, x, reward):
        # Optional: overwrite the last point if it was just suggested
        if len(self.xs) > 0 and self.xs[-1] == x:
            self.ys.append(reward)
            self.optimizer.tell(x, reward)
        else:
            # for consistency in external use
            self.xs.append(x)
            self.ys.append(reward)
            self.optimizer.tell(x, reward)


def log2_map(norm_value: float, min_val: float, max_val: float) -> float:
    """
    Maps a normalized value [0, 1] to a target range [min_val, max_val]
    using a log2-based scale.
    
    Args:
        norm_value (float): Normalized input in [0, 1]
        min_val (float): Minimum value of the target domain (must be > 0)
        max_val (float): Maximum value of the target domain (must be > 0)
    
    Returns:
        float: Mapped value in [min_val, max_val] (log₂-scaled)
    """
    assert 0.0 <= norm_value <= 1.0, "Input must be in [0, 1]"
    assert min_val > 0 and max_val > 0 and max_val > min_val, "Invalid range"
    
    log_min = math.log2(min_val)
    log_max = math.log2(max_val)
    log_val = log_min + norm_value * (log_max - log_min)
    return 2 ** log_val
