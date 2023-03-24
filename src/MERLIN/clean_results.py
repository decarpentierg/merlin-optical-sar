"""Clean results/ directory."""

from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).parent.parent.parent.absolute().as_posix()  # get path to project root
resultsdir = ROOT_DIR + "/results/"

shutil.rmtree(resultsdir + "sample/")
shutil.rmtree(resultsdir + "saved_model/")
shutil.rmtree(resultsdir + "test/")
