import argparse

from src.data.time_prior_features import TIME_FEATURE_NAMES


TIME_PRIOR_ARG_NAMES = [
    "use_temporal_prior",
    "use_prior_calibrated_logits",
    "prior_hidden_dim",
    "prior_input_dim",
    "domain_emb_dim",
    "num_domains",
    "prior_alpha",
    "prior_beta",
    "prior_dropout",
    "use_month",
    "use_season",
    "use_day_of_year",
    "use_weekday",
    "use_solar_term",
    "domain_name",
    "use_prior_guided_hard_negative",
]


def _none_or_int(value):
    """Parse integer CLI values while accepting null/none for YAML parity."""
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def add_time_prior_args(parser: argparse.ArgumentParser):
    """Add optional CLI overrides for domain-aware temporal prior settings."""
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--use_temporal_prior", action=bool_action, default=None)
    parser.add_argument("--use_prior_calibrated_logits", action=bool_action, default=None)
    parser.add_argument("--prior_hidden_dim", type=int, default=None)
    parser.add_argument("--prior_input_dim", type=_none_or_int, default=None)
    parser.add_argument("--domain_emb_dim", type=int, default=None)
    parser.add_argument("--num_domains", type=int, default=None)
    parser.add_argument("--prior_alpha", type=float, default=None)
    parser.add_argument("--prior_beta", type=float, default=None)
    parser.add_argument("--prior_dropout", type=float, default=None)
    parser.add_argument("--use_month", action=bool_action, default=None)
    parser.add_argument("--use_season", action=bool_action, default=None)
    parser.add_argument("--use_day_of_year", action=bool_action, default=None)
    parser.add_argument("--use_weekday", action=bool_action, default=None)
    parser.add_argument("--use_solar_term", action=bool_action, default=None)
    parser.add_argument("--domain_name", type=str, default=None)
    parser.add_argument("--use_prior_guided_hard_negative", action=bool_action, default=None)


def apply_time_prior_overrides(config: dict, args_cmd):
    """Apply only explicitly provided CLI values, preserving YAML defaults."""
    for name in TIME_PRIOR_ARG_NAMES:
        if hasattr(args_cmd, name):
            value = getattr(args_cmd, name)
            if value is not None:
                config[name] = value


def resolve_time_prior_config(args):
    """
    Normalize temporal-prior config after YAML/CLI parsing.

    The current data layer emits TIME_FEATURE_NAMES as a fixed-width vector.
    When prior_input_dim is null, infer it from that stable feature list before
    model construction so TemporalPriorEncoder receives a concrete dimension.
    """
    if getattr(args, "prior_input_dim", None) is None:
        args.prior_input_dim = len(TIME_FEATURE_NAMES)
        if getattr(args, "rank", 0) == 0:
            print(
                "[Time prior config] prior_input_dim was null; "
                f"inferred {args.prior_input_dim} from TIME_FEATURE_NAMES."
            )

    if getattr(args, "use_temporal_prior", False) and args.prior_input_dim <= 0:
        raise ValueError(
            "prior_input_dim must be a positive integer when "
            "use_temporal_prior=true."
        )

    return args


def print_time_prior_summary(args):
    """Print a compact startup summary for temporal-prior related settings."""
    if getattr(args, "rank", 0) != 0:
        return

    enabled_features = []
    feature_flags = [
        ("month", "use_month"),
        ("season", "use_season"),
        ("day_of_year", "use_day_of_year"),
        ("weekday", "use_weekday"),
        ("solar_term", "use_solar_term"),
    ]
    for feature_name, flag_name in feature_flags:
        if getattr(args, flag_name, False):
            enabled_features.append(feature_name)

    print(
        "[Time prior config] "
        f"use_temporal_prior={getattr(args, 'use_temporal_prior', False)}, "
        f"use_prior_calibrated_logits={getattr(args, 'use_prior_calibrated_logits', False)}, "
        f"prior_alpha={getattr(args, 'prior_alpha', None)}, "
        f"prior_beta={getattr(args, 'prior_beta', None)}, "
        f"prior_input_dim={getattr(args, 'prior_input_dim', None)}, "
        f"domain_name={getattr(args, 'domain_name', None)}, "
        f"enabled_time_features={enabled_features}"
    )
