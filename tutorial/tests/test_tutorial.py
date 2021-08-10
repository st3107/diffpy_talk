from pathlib import Path

import matplotlib.pyplot as plt
from pkg_resources import resource_filename

import tutorial.tutorial as tutorial

# turn of io to avoid the QT to block tests
plt.ioff()
# file paths
GR_FILE = resource_filename("tutorial", "data/TiO2_np_ligand.gr")
CIF_FILE = resource_filename("tutorial", "data/TiO2_bronze_mp.cif")


def test_all_functions(tmpdir):
    """Test all the functions in an example fitting."""
    base = Path(str(tmpdir))
    recipe = tutorial.create_recipe_from_files(
        "SP * TiO2",
        cif_files={"TiO2": CIF_FILE},
        functions={"SP": (tutorial.F.sphericalCF, ["r", "TiO2_size"])},
        data_file=GR_FILE,
        meta_data={"qdamp": 0.04, "qbroad": 0.02}
    )
    # recipe.show()
    steps = [["scale", "lat"], ["adp", "delta2"], ["xyz"], ["SP"]]
    tutorial.optimize_params(recipe, steps, rmin=1.6, rmax=10., rstep=0.1, ftol=1e-3)
    # recipe.show()
    tutorial.visualize_fits(recipe)
    tutorial.save_results(recipe, str(base), "test", pg_names=["TiO2"])
    # print("\n".join(map(str, base.glob("*"))))
    fgr_file = base.joinpath("test.fgr")
    dst_file = base.joinpath("test_diff.gr")
    tutorial.export_diff_from_fgr(str(fgr_file), str(dst_file))
    # print(np.loadtxt(str(dst_file)))
