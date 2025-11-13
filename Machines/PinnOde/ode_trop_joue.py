import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx


#==================================================================
def plot_solutions(t_plot, u_pred, u_true, u_cg, t_colloc, base_dict=None):
    if base_dict is not None:
        plt.subplot(211)
    plt.plot(t_plot, u_pred, '-', label='app')
    if u_true is not None:
        plt.plot(t_plot, u_true, ':', label='sol')
    if u_cg is not None:
        plt.plot(t_colloc, u_cg, ':', label='cg')
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='t_colloc')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(r"$u' = f(u)$")
    plt.grid()
    if base_dict is not None:
        plt.subplot(212)
        plt.title(r"bases")
        plt.plot(t_plot, base_dict.T)
        plt.xlabel("t")
        plt.grid()
    plt.show()

#==================================================================
def swish(x):
    return x * jax.nn.sigmoid(x)
#==================================================================
class Machine:
    def __init__(self, layers):
        self.layers = [1] * (len(layers)+1)
        self.layers[1:] = layers
    def init_params(self):
        key = jax.random.PRNGKey(0)
        params = []
        for l in range(1, len(self.layers)):
            key, subkey = jax.random.split(key)
            W = jax.random.normal(subkey, (self.layers[l], self.layers[l-1])) * np.sqrt(2 / self.layers[l-1])
            b = np.zeros(self.layers[l])
            params.append((W, b))
        return params
    def bases(self, params, t):
        t = np.atleast_1d(t)   # shape (1,)
        x = t[None, :]          # shape (1, 1)
        for W, b in params[:-1]:
            # x = swish(W @ x + b[:, None])
            x = np.tanh(W @ x + b[:, None])
        W, b = params[-1]
        return W @ x + b[:, None]
    def regularization(self, params, t_colloc):
        M = self.bases(params, t_colloc)
        e = np.sum(M, axis=0)-1
        # K = M.T @ M
        # reg_gram = np.mean((K - np.eye(K.shape[0])) ** 2)
        # M = np.minimum(M,0)
        return np.mean(e ** 2)# + np.mean(M*M)# + reg_gram

#==================================================================
class ModelEdo:
    def __init__(self, app, machine, n_colloc):
        self.app, self.machine = app, machine
        self.nbases = machine.layers[-1]
        self.nout = 1 if type(app.x0)==float else len(app.x0)
        self.t_colloc =  np.linspace(app.t_begin, app.t_end, n_colloc)
    def init_params(self):
        params_mach = self.machine.init_params()
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (self.nout, self.nbases)) * np.sqrt(2 / self.nbases)
        b = np.zeros(self.nout)
        coeff = W
        return params_mach, coeff
    def forward(self, params_mach, coeff, t):
        x = self.machine.bases(params_mach, t)
        W = coeff
        out = W @ x# + b[:, None]
        return out.squeeze()
    def residual_edo_single(self, params_mach, coeff, t):
        u = self.forward(params_mach, coeff, t)
        dudt = jax.jacrev(self.forward, argnums=2)(params_mach, coeff, t)
        return dudt - self.app.f(u)
    def residual_edo(self, params_mach, coeff):
        return jax.vmap(lambda t: self.residual_edo_single(params_mach, coeff, t))(self.t_colloc)
    def residual_bdry(self, params_mach, coeff):
        return self.forward(params_mach, coeff, self.t_colloc[0])-self.app.x0
    def loss(self, params_mach, coeff):
        res_dom = self.residual_edo(params_mach, coeff)
        res_bdry = self.residual_bdry(params_mach, coeff)
        return np.mean(res_dom ** 2) + np.mean(res_bdry ** 2)

def train_bases(params_machine, machine, t_colloc, lr=0.1, n_epochs=100):
    optimizer = optax.lbfgs(learning_rate=lr)
    opt_state = optimizer.init(params_machine)
    @jax.jit
    def train_step(params, opt_state):
        loss = lambda p: machine.regularization(p, t_colloc)
        loss_value, grads = jax.value_and_grad(loss)(params)
        # updates, opt_state = optimizer.update(grads, opt_state, params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    for epoch in range(n_epochs):
        params_machine, opt_state, loss_value = train_step(params_machine, opt_state)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")
    return params_machine
def train_solve(coefs, model, params_machine, lr=0.1, n_epochs=100):
    optimizer = optax.lbfgs(learning_rate=lr)
    opt_state = optimizer.init(coefs)
    @jax.jit
    def train_step(params, opt_state):
        loss = lambda p: model.loss(params_machine, p)
        loss_value, grads = jax.value_and_grad(loss)(params)
        # updates, opt_state = optimizer.update(grads, opt_state, params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss)
        # print(f"{params=} {grads=} {updates=}")
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    for epoch in range(n_epochs):
        coefs, opt_state, loss_value = train_step(coefs, opt_state)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")
    return coefs
def solve_ls(coefs, model, params_machine):
    shape = coefs.shape
    coefs = np.array(coefs).flatten()
    from scipy.optimize import least_squares
    def f(p):
        pc = p.reshape(shape)
        res_dom = model.residual_edo(params_machine, pc).flatten()
        res_bdry = model.residual_bdry(params_machine, pc).flatten()
        # print(f"{res_dom.shape=} {res_bdry.shape=} {res_bdry=}")
        return np.array(np.concatenate([res_dom, res_bdry]))
    # print(f"{coefs.shape=} {coefs=}")
    result = least_squares(f, coefs)
    print(f"{result.message=} {result.status=} {result.nfev=}")
    return np.array(result.x.reshape(shape))

def train_all(params, model, lr=0.01, n_epochs=1000):
    # optimizer = optax.adam(learning_rate=lr)
    optimizer = optax.lbfgs(learning_rate=lr)
    opt_state = optimizer.init(params)
    def loss(params):
        params_mach, coeff = params
        return model.loss(params_mach, coeff)
    @jax.jit
    def train_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        # updates, opt_state = optimizer.update(grads, opt_state, params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for epoch in range(n_epochs):
        params, opt_state, loss_value = train_step(params, opt_state)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")
    return params


if __name__ == '__main__':
    # Application
    import ode_examples
    app, layers, n_colloc = ode_examples.Exponential(), [3,3], 6
    # app, layers, n_colloc = ode_examples.Exponential(), [11,11], 12
    # app, layers, n_colloc = ode_examples.Exponential(), [23,23], 24
    # app, layers, n_colloc = ode_examples.Pendulum(t_end=3.5), [24,24], 25
    # Machine setup
    # Collocation points
    machine = Machine(layers)
    model = ModelEdo(app, machine, n_colloc)
    params_machine, coefs = model.init_params()


    params_machine = train_bases(params_machine, machine, model.t_colloc)
    # coefs = train_solve(coefs, model, params_machine, lr=0.01, n_epochs=100)
    coefs = solve_ls(coefs, model, params_machine)
    # params_machine, coefs = train_all((params_machine, coefs), model, 0.001, n_epochs=3000)
    # print(f"coefs={coefs}")
    # coefs = solve_ls(coefs, model, params_machine)
    # print(f"coefs={coefs}")

    # Plot results
    t_plot = np.linspace(model.t_colloc[0], model.t_colloc[-1], 200)
    u_pred = jax.vmap(lambda t: model.forward(params_machine, coefs, t))(t_plot)
    if hasattr(app, 'solution'):
        u_true = app.solution(t_plot)
        u_cg = None
        err = np.mean((u_true-u_pred)**2)
    else:
        u_true = None
        import ode_solver
        cgp = ode_solver.CgK(k=2)
        u_node, u_coef = cgp.run_forward(model.t_colloc, app)
        u_cg = u_node.T
        u_machine = jax.vmap(lambda t: model.forward(params_machine, coefs, t))(model.t_colloc)
        err = np.mean((u_cg-u_machine)**2)
    print(f"err = {err:.5e}")

    base_dict = model.machine.bases(params_machine, t_plot)
    plot_solutions(t_plot, u_pred, u_true, u_cg, model.t_colloc, base_dict=base_dict)