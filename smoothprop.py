class SmoothProp(Optimizer):
    def __init__(self, lr=1e-4, beta=0.9, l2=1e-5, epsilon=1e-6, *args, **kwargs):
        super(SmoothProp, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.0)
        self.lr = K.variable(lr)
        self.l2 = l2
        self.beta = K.variable(beta)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1)]
        
        t = self.iterations + 1
        ag_comp = 1.-K.exp(-t*(1.0-self.beta)/self.beta)

        for p, g, c in zip(params, grads, constraints):
            # filtered gradient
            sg = K.variable(np.zeros(K.get_value(p).shape))
            ag = K.variable(np.zeros(K.get_value(p).shape))
            
            # l2 penalty
            if self.l2 > 0.:
                g = g + self.l2*p
            
            # update gradient filter
            filter = lambda beta, x0, x1: beta*x0 + (1.-beta)*x1
            sg_t = filter(self.beta, sg, g)
            ag_t = filter(self.beta, ag, K.abs(g))
            
            step = sg_t * (ag_comp / (ag_t + self.epsilon))
            
            # update parameter
            p_t = p - self.lr*step

            self.updates.append((sg, sg_t))
            self.updates.append((ag, ag_t))
            self.updates.append((p, c(p_t)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "l2": float(K.get_value(self.l2)),
                "beta": float(K.get_value(self.beta)),
                "epsilon": self.epsilon}