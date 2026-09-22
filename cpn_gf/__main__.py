from __future__ import annotations

import argparse

from .analysis import analyze_path
from .recommend import print_recommendation, recommend_chains
from .runner import pilot_config, pilot_experiment, resume_run, run_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m cpn_gf")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start a new experiment")
    run.add_argument("--config", required=True)
    pilot = sub.add_parser("pilot", help="run only scale-selection pilots")
    pilot_source = pilot.add_mutually_exclusive_group(required=True)
    pilot_source.add_argument("--config")
    pilot_source.add_argument("--run")
    resume = sub.add_parser("resume", help="resume a mul run or extend an experiment")
    resume.add_argument("--run", required=True)
    analyze = sub.add_parser("analyze", help="rebuild aggregate results")
    analyze.add_argument("--run", required=True)
    recommend = sub.add_parser("recommend-chains", help="benchmark and recommend HMC chains")
    recommend.add_argument("--config", required=True)
    size = recommend.add_mutually_exclusive_group()
    size.add_argument("--lattice-size", type=int)
    size.add_argument("--pilot", action="store_true")
    recommend.add_argument("--max-chains", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.command == "run":
        path, _ = run_config(args.config)
        print(path)
    elif args.command == "pilot":
        result = (pilot_config(args.config) if args.config
                  else pilot_experiment(args.run))
        print(result["experiment"])
        for item in result["runs"]:
            values = item.get("pilot") or {}
            xi = values.get("xi")
            detail = ("" if xi is None else
                      f" xi={xi:.8g} +/- {values.get('xi_error', float('nan')):.3g}"
                      f" recommended_L={values.get('recommended_L', values.get('L'))}")
            print(f"mul={item['mul']:g}: {item['action']} ({item['status']}){detail}")
    elif args.command == "resume":
        result = resume_run(args.run)
        if "runs" in result:
            for item in result["runs"]:
                print(f"mul={item['mul']:g}: {item['action']} ({item['status']})")
        else:
            print(result["status"])
    elif args.command == "analyze":
        print(analyze_path(args.run))
    else:
        print_recommendation(recommend_chains(
            args.config, lattice_size=args.lattice_size,
            run_pilot=args.pilot, max_chains=args.max_chains))


if __name__ == "__main__":
    main()
