import torch
import torch.nn.functional as F
from copy import deepcopy
import math
import logging


class FedOda:
    def __init__(self, model_fn, device, pooled_loader, total_clients_num, 
                 malicious_clients_num, E=5, learning_rate=0.01, scaling_factor=8,
                 reduce_factor=0.8, increase_factor=1.1, threshold=0.7, logger=None):
        self.model_fn = model_fn
        self.device = device
        self.E = E # Number of rounds to consider for history
        self.learning_rate = learning_rate
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
        grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        model.zero_grad()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            break  # One batch is sufficient for direction

        grad_flattened = []
        for name, param in model.named_parameters():
            g = param.grad.detach().clone()
            grads[name] = g
            grad_flattened.append(g.view(-1))

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
            break  # One batch

        grad_flattened = []
        for param in model.parameters():
            grad_flattened.append(param.grad.view(-1))
        grad_vector = torch.cat(grad_flattened)

        update_norm = torch.norm(flatten_model(model_t.state_dict()) - flatten_model(model_t_minus_1.state_dict()))
        benign_direction = grad_vector / (grad_vector.norm() + 1e-10)
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
            self.logger.warning("[FedOda] No history for Bayesian inference. Returning neutral posterior.")
            return 0.5

        # Compute empirical variance between server deltas and predicted benign deltas
        sigma2_numerator = sum(
            (self.history_server_updates[i] - self.history_pred_benign_updates[i]).norm()**2
            for i in range(E)
        )
        sigma2 = sigma2_numerator / (d * E + 1e-10)

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
                f"[FedOda] Iter {i}: "
                f"||Δs - Δb||² = {d1:.6f}, "
                f"||Δs - Δs_pred||² = {d2:.6f}, "
                f"log-likelihood diff = {l:.6f}"
            )

            log_likelihoods.append(l)

        total_ll = sum(log_likelihoods)
        posterior = 1 / (1 + math.exp(-total_ll))

        self.logger.info(f"[FedOda] Bayesian total log-likelihood = {total_ll:.6f}")
        self.logger.info(f"[FedOda] Posterior P(H1|D) = {posterior:.4f}")

        return posterior

    def adjust_epsilon(self, epsilon, posterior):
        if posterior < self.threshold:
            return epsilon * self.reduce_factor
        else:
            return epsilon
        
    def attack(self):
        if self.server_model is None:
            self.logger.warning("[FedOda] Server model is None. Cannot proceed with attack.")
            return None, None

        # 🔁 First Iteration: No history
        if self.previous_server_model is None:
            self.logger.info("[FedOda] First iteration detected. Performing special attack initialization.")

            for x, y in self.pooled_loader:
                print(f"x shape: {x.shape}, y: {y}")
                break

            # Step 1: Compute attack direction (g_m)
            g_m_dict, g_m_vec = self.get_attack_direction(self.server_model, self.pooled_loader)

            # Step 2: Simulate benign update from one malicious client
            benign_update_vec, benign_norm = self.simulate_benign_update(self.pooled_loader, max_batches=10)
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
        print(f"self.epsilon: {self.epsilon}")
        
        g_m_dict, g_m_vec = self.get_attack_direction(self.server_model, self.pooled_loader)
        malicious_update = self.epsilon * g_m_vec

        self.logger.info("[FedOda] Step 2: Predicting benign update direction")
        benign_update = self.compute_predicted_benign_update(self.server_model, self.previous_server_model, self.pooled_loader)

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
        self.logger.info(f"[FedOda] Posterior P(H1|D) = {posterior:.4f}")

        self.logger.info("[FedOda] Step 8: Adjusting scaling factor (epsilon)")
        prev_epsilon = self.epsilon
        print("prev_epsilon", prev_epsilon)
        self.epsilon = self.adjust_epsilon(prev_epsilon, posterior)
        self.logger.info(f"[FedOda] Epsilon L2 adjusted: {prev_epsilon:.4f} -> {self.epsilon:.4f}")

        self.logger.info("[FedOda] Step 9: Creating malicious update dict")
        malicious_update_dict = vector_to_state_dict(malicious_update, self.server_model)

        self.logger.info("[FedOda] Attack completed, returning malicious update.")
        return malicious_update_dict, malicious_update


    def simulate_benign_update(self, loader, max_batches=1):
        """
        Simulate benign training to estimate the update norm for one client.
        
        Args:
            loader: DataLoader containing malicious client data.
            max_batches: Number of batches to simulate (usually 1 for speed).
        
        Returns:
            update_vector: Flattened parameter update vector.
            update_norm: L2 norm of the update vector.
        """
        model_before = deepcopy(self.server_model)
        model_after = deepcopy(self.server_model).to(self.device)
        model_after.train()
        optimizer = torch.optim.SGD(model_after.parameters(), lr=self.learning_rate)

        batch_count = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model_after(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_count += 1
            if batch_count >= max_batches:
                break

        # Compute flattened update vector
        before = flatten_model(model_before.state_dict())
        after = flatten_model(model_after.state_dict())
        update = after - before
        norm = update.norm()

        return update, norm


# Utility
def flatten_model(state_dict):
    return torch.cat([param.view(-1).float() for param in state_dict.values()])

def vector_to_state_dict(vector, reference_model):
    """
    Convert a flattened vector into a state_dict that matches the structure of reference_model.

    Args:
        vector (torch.Tensor): Flattened model update vector.
        reference_model (torch.nn.Module): A model whose parameter shapes are used for reconstruction.

    Returns:
        state_dict (dict): A dictionary of parameter tensors shaped like reference_model.
    """
    state_dict = {}
    idx = 0
    for name, param in reference_model.named_parameters():
        numel = param.numel()
        state_dict[name] = vector[idx: idx + numel].view_as(param).clone().detach()
        idx += numel
    return state_dict
