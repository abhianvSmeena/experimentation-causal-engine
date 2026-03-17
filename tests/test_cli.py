from experiment_toolkit.cli import main


def test_cli_runs_end_to_end(capsys):
    main(["--n", "20000", "--lift", "0.05", "--seed", "1"])
    out = capsys.readouterr().out
    for heading in ["Design", "Sample ratio", "Fixed-horizon", "CUPED", "Sequential", "Bayesian"]:
        assert heading in out
