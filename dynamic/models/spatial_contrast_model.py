from torch import nn


class SCModel(nn.Module):
    def __init__(self, sc_filter, t_filter, crop):
        super().__init__()
        # sc_filter, t_filter = regularize(sc_filter, t_filter)
        # self.filters =
        core = None  # Factorized, init with sc, t filters, specify the filter size based on the crop

        readout = None  # Identity
        # self.parameters

    def forward(self, x):
        # i_mean = sc, t filters.dot(x)
        # lsc = sc, t filters.dot(x but more complicated)
        # out = parameter_w * (i_mean + sc)
        # out = readout(out)
        # out = parameter_a * out+ parameter_b
        # return out
        pass

    def regularize(self, sc_filter, t_filter):
        smoothened_filters = None
        return smoothened_filters

    @staticmethod
    def build_initial(dataloaders, sta_location, cell_index):
        # stas = load from wherever
        # filters = get from stas
        # crop = whaterver dataloader says
        # model = SCModel(filter)
        # assign filters to weights
        # requires_grad = False
        # return model
        pass

    def allow_training(self):
        # params require grad
        pass


# run_scmodel script - takes parameters from command line
# "train_sc_model" - is called by run scmodel_script, initialized model and dataloaders, calls train() function
