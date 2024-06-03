from src.parsers.configs.generic import DEFAULT_ARGS_ALLO, DEFAULT_ARGS_EGO

DEFAULT_ARGS_EGO["img_feat_version"] = ""  # should use ArcticDataset
DEFAULT_ARGS_ALLO["img_feat_version"] = ""  # should use ArcticDataset
DEFAULT_ARGS_EGO["batch_size"] = 50
DEFAULT_ARGS_EGO["test_batch_size"] = 50
DEFAULT_ARGS_ALLO["batch_size"] = 50
DEFAULT_ARGS_ALLO["test_batch_size"] = 50