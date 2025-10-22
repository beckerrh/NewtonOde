import jax.numpy as jnp
import jax
import optax
from flax import nnx

#-----------------------------------------------------------
def train_machine_and_solve(machine, ode_loss):
    graphdef, params, batch_stats = nnx.split(machine, nnx.Param, nnx.BatchStat)
    def loss(params):
        # ode loss
        machine_tmp = nnx.merge(graphdef, params, batch_stats)
        return ode_loss(machine_tmp)

    optimizer = optax.lbfgs(learning_rate=0.1)
    opt_state = optimizer.init(params)
    @jax.jit
    def train_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads, updates

    n_epochs = 1200
    for epoch in range(n_epochs):
        params, opt_state, loss_value, grads, updates = train_step(params, opt_state)
        if epoch % 100== 0:
            print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")
    return nnx.merge(graphdef, params, batch_stats)


#-----------------------------
def split_feature_coeff(params):
    """
    Split the parameters into feature layers and coefficient layer.
    Assumes the last Linear layer is the coefficient layer.
    """
    params_feat = {}
    params_coeff = {}
    for name, value in params.items():
        # print(f"@{name=}")
        if name == "final_layer":
            params_coeff[name] = value
        else:
            params_feat[name] = value
    return params_feat, params_coeff


def train_features(machine, machine_loss, n_epochs=101, lr=0.01):
    # split machine into params and graph
    graphdef, params, batch_stats = nnx.split(machine, nnx.Param, nnx.BatchStat)

    # separate feature vs coefficient params
    params_feat, params_coeff = split_feature_coeff( params)

    optimizer = optax.adam(lr)
    # optimizer = optax.lbfgs(lr)
    opt_state = optimizer.init(params_feat)

    def loss_feat(params_feat):
        # merge params back with fixed coeff_layer
        params_full = {**params_feat, **params_coeff}  # merge PyTrees manually
        machine_tmp = nnx.merge(graphdef, params_full, batch_stats)
        return machine_loss(machine_tmp)

    @jax.jit
    def train_step(params_feat, opt_state):
        loss_val, grads = jax.value_and_grad(loss_feat)(params_feat)
        # updates, opt_state = optimizer.update(grads, opt_state, params_feat)
        updates, opt_state = optimizer.update(
            grads, opt_state, params_feat,
            value=loss_val, grad=grads, value_fn=loss_feat
        )
        params_feat = optax.apply_updates(params_feat, updates)
        return params_feat, opt_state, loss_val

    for epoch in range(n_epochs):
        params_feat, opt_state, loss_val = train_step(params_feat, opt_state)
        if epoch % 50 == 0:
            print(f"-Basis- Epoch {epoch:4d}, Loss: {loss_val:.3e}")
    # merge back full machine with trained features
    params_full = {**params_feat, **params_coeff}
    trained_machine = nnx.merge(graphdef, params_full, batch_stats)
    return trained_machine
#-----------------------------
def solve_coefficients(machine, ode_loss, n_epochs=400, lr=0.1):
    """
    Train only the final layer (coefficients) of the machine
    to satisfy the collocated ODE residual.
    """

    # Split machine into graph, params, batch_stats
    graphdef, params, batch_stats = nnx.split(machine, nnx.Param, nnx.BatchStat)

    # Find the final layer name
    # final_layer_name = [name for name in params if 'final_layer' in name][0]
    final_layer_name = 'final_layer'
    final_params = params[final_layer_name]

    # Extract JAX arrays from VariableState
    W0 = final_params['kernel'].value  # shape (out_features, in_features)
    b0 = final_params['bias'].value    # shape (out_features,)

    # Flatten to 1D vector for optimizer
    coeff_init = jnp.concatenate([W0.ravel(), b0.ravel()])

    # Helper to reconstruct full params tree with new coefficients
    def rebuild_params(coeff_vec):
        n_out, n_in = W0.shape
        W_new = nnx.Param(coeff_vec[:n_out * n_in].reshape((n_out, n_in)))
        b_new = nnx.Param(coeff_vec[n_out * n_in:])
        # nnx.State can be built from dict of dicts
        new_params_dict = {**params._mapping}  # copy underlying dict
        new_params_dict[final_layer_name] = {"kernel": W_new, "bias": b_new}
        return nnx.State(new_params_dict)  # build new State

    # Loss function: collocated ODE residual only
    def loss_fn(coeff_vec):
        new_params = rebuild_params(coeff_vec)
        machine_tmp = nnx.merge(graphdef, new_params, batch_stats)
        return ode_loss(machine_tmp)

    # LBFGS optimizer from Optax
    optimizer = optax.lbfgs(lr)
    # optimizer = optax.adam(0.01*lr)
    opt_state = optimizer.init(coeff_init)

    @jax.jit
    def step(coeff_vec, opt_state):
        loss_val, grads = jax.value_and_grad(loss_fn)(coeff_vec)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, coeff_vec,
            value=loss_val, grad=grads, value_fn=loss_fn
        )
        coeff_vec = optax.apply_updates(coeff_vec, updates)
        return coeff_vec, opt_state, loss_val

    coeff_vec = coeff_init
    for epoch in range(n_epochs):
        coeff_vec, opt_state, loss_val = step(coeff_vec, opt_state)
        if epoch % 50 == 0:
            print(f"-Coeff- Epoch {epoch:4d}, Loss={loss_val:.3e}")

    # Merge optimized coefficients back
    new_params = rebuild_params(coeff_vec)
    trained_machine = nnx.merge(graphdef, new_params, batch_stats)
    return trained_machine

