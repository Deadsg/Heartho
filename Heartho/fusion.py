class KnowledgeFusionEngine:
    def __init__(self, strategy="first"):
        self.strategy = strategy

    def fuse(self, results):
        if self.strategy == "first":
            return list(results.values())[0]
        return "Fusion strategy not implemented"
