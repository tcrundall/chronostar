import chronostar

config_data = sys.arg[1]

data=...
if config.convert:
    cartesian=chronostar.convert(data, output=...)

perform_inc_em_fit(data, config_file)
ˀ

# ------ vs --------
chronostar.perform_fit(config_file)
