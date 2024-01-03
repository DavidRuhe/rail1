class Namespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = Namespace(**v)
            setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        if isinstance(v, dict):
            v = Namespace(**v)
        setattr(self, k, v)
        return self

    def __setattr__(self, k, v):
        if isinstance(v, dict):
            v = Namespace(**v)
        return super().__setattr__(k, v)

    def __repr__(self):
        return str({k: getattr(self, k) for k in dir(self) if not k.startswith("__")})
