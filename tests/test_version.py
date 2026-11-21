import eflomalr
from eflomalr import _input
from eflomalr.cli import main


def test_version():
    assert isinstance(eflomalr.__version__, str) and eflomalr.__version__


def test_cli_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"eflomalr {eflomalr.__version__}\n"


def test_input_normalisation():
    assert _input.sentence_to_line("a  b\n", "x") == "a b"
    assert _input.text_source([["a", "b"], "c d"], "src") == ["a b", "c d"]
