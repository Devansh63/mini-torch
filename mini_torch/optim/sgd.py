class SGD:
    """
    Stochastic Gradient Descent optimizer.
    Update rule: p.data -= lr * p.grad

    Usage:
        opt = SGD(model.parameters(), lr=1e-2)
        opt.zero_grad()   # clear accumulated gradients
        loss.backward()   # compute gradients
        opt.step()        # apply parameter update
    """
    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = float(lr)

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data = (p.data - self.lr * p.grad).astype("float32")

    def zero_grad(self):
        for p in self.params:
            p.grad = None
