import torch
import torch.nn.functional as F
from copy import deepcopy
import math
import logging
from torch.utils.data import DataLoader

class FedOda:
    def __init__(self, model_fn, device, pooled_loader, total_clients_num, 
                 malicious_clients_num, E=5, scaling_factor=8,
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
        self.total_clients_num = total_clients_num
        self.malicious_clients_num = malicious_clients_num
        self.current_s_m = None
        
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
                
        # Logging
        self.logger = logger or logging.getLogger("FedOda")

        # Historical memory
        self.history_server_updates = []
        self.history_pred_benign_updates = []
        self.history_pred_server_updates = []
        self.history_s_m = []

    def get_attack_direction(self, model, train_loader):
        model.train()
        loss_fn = lambda outputs, labels: -F.cross_entropy(outputs, labels)

        model.zero_grad()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            break

        grads = {}
        model_sd = model.state_dict()
        named_param_dict = dict(model.named_parameters())

        for name in model_sd:
            if name in named_param_dict:
                param = named_param_dict[name]
                if param.grad is not None:
                    grads[name] = param.grad.detach().clone()
                else:
                    grads[name] = torch.zeros_like(param)
            else:
                # For buffers like running_mean, running_var: zero them out or skip
                grads[name] = torch.zeros_like(model_sd[name])

        # Flatten
        grad_flattened = [g.view(-1) for g in grads.values()]
        g_m_vec = torch.cat(grad_flattened)
        g_m_unit = g_m_vec / (g_m_vec.norm() + 1e-10)
        self.att_direction = g_m_unit
        return grads, g_m_unit


    def compute_predicted_benign_update(self, model_t, model_t_minus_1, train_loader):
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
        history_len = len(self.history_s_m)
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

    def update_history(self, observed_server_update, predicted_benign, predicted_server, s_m):
        self.history_server_updates.append(observed_server_update)
        self.history_pred_benign_updates.append(predicted_benign)
        self.history_pred_server_updates.append(predicted_server)
        self.history_s_m.append(s_m) # selection number of malicious clients in each iteration 

        if len(self.history_server_updates) > self.E:
            self.history_server_updates.pop(0)
            self.history_pred_benign_updates.pop(0)
            self.history_pred_server_updates.pop(0)
            self.history_s_m.pop(0)

    def bayesian_inference(self, d):
        E = len(self.history_server_updates)

        if E == 0:
            self.logger.warning("[FedOda] No history found for Bayesian inference. Returning neutral posterior.")
            return 0.5

        # Compute sigma² (empirical variance)
        sigma2 = sum(
            (self.history_server_updates[i] - self.history_pred_benign_updates[i]).norm()**2
            for i in range(E)
        ) / (d * E + 1e-10)

        self.logger.info(f"[FedOda] Bayesian inference: history length E = {E}")
        self.logger.info(f"[FedOda] sigma² = {sigma2:.6f}")

        log_likelihoods = []
        for i in range(E):
            delta_s = self.history_server_updates[i]
            delta_b = self.history_pred_benign_updates[i]
            delta_s_pred = self.history_pred_server_updates[i]

            d1 = (delta_s - delta_b).norm()**2
            d2 = (delta_s - delta_s_pred).norm()**2
            l = (d1 - d2) / (2 * sigma2 + 1e-10)

            self.logger.info(
                f"[FedOda] Iter: t-{i}: "
                f"||Δs - Δb||² = {d1:.6f}, "
                f"||Δs - Δs_pred||² = {d2:.6f}, "
                f"log-likelihood diff = {l:.6f}"
            )

            log_likelihoods.append(l)

        # ✅ Use average log-likelihood
        avg_ll = sum(log_likelihoods) / E

        # 🔒 Clamp to prevent exp overflow
        max_exp = 80
        clamped_ll = max(min(avg_ll, max_exp), -max_exp)

        # ✅ Tune temperature
        temperature = 5 + 1000 * sigma2
        posterior = 1 / (1 + math.exp(-clamped_ll / temperature))

        self.logger.info(f"[FedOda] Bayesian avg log-likelihood = {avg_ll:.6f} (clamped: {clamped_ll:.2f})")
        self.logger.info(f"[FedOda] Posterior P(H1|D) = {posterior:.4f}")
        return posterior


    def adjust_epsilon(self, epsilon, posterior):
        if posterior < self.threshold:
            return epsilon * self.reduce_factor
        else:
            return epsilon * self.increase_factor

    def adjust_scaling_factor(self, scaling_factor, posterior):
        """
        Adjusts the scaling factor based on the history of posterior probabilities.
        Keeps it stable if posterior is close to the threshold.

        Returns:
            float: Updated and clamped scaling factor.
        """
        # Save posterior history
        if not hasattr(self, "posterior_history"):
            self.posterior_history = []
        self.posterior_history.append(posterior)
        if len(self.posterior_history) > self.E:
            self.posterior_history.pop(0)

        # Compute average posterior over last E rounds
        avg_posterior = sum(self.posterior_history) / len(self.posterior_history)

        self.logger.info(f"[FedOda] Avg posterior over last {len(self.posterior_history)} rounds = {avg_posterior:.4f}")

        # Stability margin around threshold
        margin = 0.1
        lower = self.threshold - margin
        upper = self.threshold + margin

        if lower <= avg_posterior <= upper:
            # Posterior is close to threshold — keep it stable
            self.logger.info("[FedOda] Posterior near threshold — scaling factor unchanged.")
            new_factor = scaling_factor
        elif avg_posterior < self.threshold:
            new_factor = scaling_factor * self.reduce_factor
            self.logger.info("[FedOda] Posterior low — increasing stealth (reduce scaling).")
        else:
            new_factor = scaling_factor * self.increase_factor
            self.logger.info("[FedOda] Posterior high — increasing attack power.")

        # Clamp to [0.5, 20]
        new_factor = max(0.5, min(20.0, new_factor))
        return new_factor

        
    def attack(self):
        if self.server_model is None:
            self.logger.warning("[FedOda] Server model is None. Cannot proceed with attack.")
            return None, None

        # 🔁 First Iteration: No history
        if self.previous_server_model is None:
            self.logger.info("[FedOda] First iteration detected. Performing special attack initialization.")

            # for x, y in self.pooled_loader:
            #     # print(f"x shape: {x.shape}, y: {y}")
            #     break

            # Step 1: Compute attack direction (g_m)
            g_m_dict, g_m_vec = self.get_attack_direction(self.server_model, self.pooled_loader)

            # Step 2: Simulate benign update from one malicious client
            benign_update_vec, benign_norm = self.simulate_benign_update(self.pooled_loader,
                                                                         data_size=self.client_data_size,
                                                                         batch_size=self.client_batch_size)
            self.epsilon = self.scaling_factor * benign_norm
            self.logger.info(f"[FedOda] Simulated benign norm = {benign_norm:.4f}, ε = {self.epsilon:.4f}")

            # Step 3: Apply attack
            malicious_update = self.epsilon * g_m_vec

            # Step 4: Convert vector to dict
            malicious_update_dict = vector_to_state_dict(malicious_update, self.server_model)
            self.logger.info(f"[FedOda] First-round attack completed. Returning malicious update.")
            return malicious_update_dict, malicious_update

        # 🔁 Regular FedOda Attack
        self.logger.info("[FedOda] Step 1: Computing attack direction (g_m)")
        g_m_dict, g_m_vec = self.get_attack_direction(self.server_model, self.pooled_loader)
        
        self.logger.info("[FedOda] Step 2: Predicting benign update direction")
        benign_update = self.compute_predicted_benign_update(self.server_model, 
                                                             self.previous_server_model, 
                                                             self.pooled_loader)
        benign_norm = benign_update.norm()
        self.epsilon = self.scaling_factor * benign_norm
        malicious_update = self.epsilon * g_m_vec
        self.logger.info(f"[FedOda] benign_norm = {benign_norm:.4f}, self.scaling_factor = {self.scaling_factor:.4f}, ε = {self.epsilon:.4f}")

        self.logger.info("[FedOda] Step 3: Estimating benign client count (s_b)")
        s_b, s_hat = self.estimate_benign_clients(
            total_clients_num=self.total_clients_num,
            malicious_clients_num=self.malicious_clients_num,
            current_s_m=self.current_s_m
        )
        self.logger.info(f"[FedOda] Estimated s_b = {s_b:.2f}, s_hat = {s_hat:.2f}")

        self.logger.info("[FedOda] Step 4: Estimating server update assuming malicious inclusion")
        delta_s_pred = self.estimate_server_update(s_b, self.current_s_m, benign_update, malicious_update)

        self.logger.info("[FedOda] Step 5: Observing actual server update")
        delta_s = flatten_model(self.server_model.state_dict()) - flatten_model(self.previous_server_model.state_dict())

        self.logger.info("[FedOda] Step 6: Updating history")
        self.update_history(delta_s, benign_update, delta_s_pred, self.current_s_m)

        self.logger.info("[FedOda] Step 7: Performing Bayesian inference")
        posterior = self.bayesian_inference(d=len(malicious_update))

        # ✅ Store posterior in history
        if not hasattr(self, "posterior_history"):
            self.posterior_history = []
        self.posterior_history.append(posterior)
        if len(self.posterior_history) > self.E:
            self.posterior_history.pop(0)

        self.logger.info("[FedOda] Step 8: Adjusting scaling factor")
        prev_factor = self.scaling_factor
        self.scaling_factor = self.adjust_scaling_factor(prev_factor, posterior)
        self.logger.info(f"[FedOda] Scaling factor adjusted: {prev_factor:.4f} -> {self.scaling_factor:.4f}")

        self.logger.info("[FedOda] Step 9: Creating malicious update dict")
        malicious_update_dict = vector_to_state_dict(malicious_update, self.server_model)

        self.logger.info(f"[FedOda] Posterior P(H1|D) = {posterior:.4f}, new scaling_factor = {self.scaling_factor:.4f}, ε recomputed = {self.scaling_factor * benign_norm:.4f}")
        self.logger.info("[FedOda] Attack completed, returning malicious update.")
        
        return malicious_update_dict, malicious_update


    def simulate_benign_update(self, loader, data_size=500, batch_size=64):
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

        return update, norm




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

