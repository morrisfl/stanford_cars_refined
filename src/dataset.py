from torchvision.datasets import ImageFolder


class VCoRDataset(ImageFolder):
    def __init__(self, root, transform, excluded_cls):
        super().__init__(root, transform)
        excluded_idx = []
        for cls_name in excluded_cls:
            excluded_idx.append(self.class_to_idx[cls_name])
            self.classes.remove(cls_name)
            self.class_to_idx.pop(cls_name)

        old_to_new_idx = {old: new for new, old in enumerate(self.class_to_idx.values())}

        samples = []
        targets = []
        for sample in self.samples:
            if not sample[1] in excluded_idx:
                new_idx = old_to_new_idx[sample[1]]
                samples.append((sample[0], new_idx))
                targets.append(new_idx)

        self.samples = samples
        self.targets = targets
        self.class_to_idx = {cls_name: old_to_new_idx[idx] for cls_name, idx in self.class_to_idx.items()}