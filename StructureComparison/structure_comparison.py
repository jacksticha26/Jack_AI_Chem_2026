import tkinter as tk
from tkinter import filedialog
from Bio.PDB import PDBParser, Superimposer, PDBIO
import numpy as np
import matplotlib.pyplot as plt
import nglview as nv


# ----------------------------
# File selection
# ----------------------------
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title)


# ----------------------------
# Load structure
# ----------------------------
def load_structure(path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(path, path)


# ----------------------------
# CA extraction + pLDDT
# ----------------------------
def extract_ca_and_plddt(structure):
    ca_atoms = []
    residues = []
    plddt = []

    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    ca_atoms.append(res["CA"])
                    residues.append(res)

                    # AlphaFold: B-factor = pLDDT
                    plddt.append(res["CA"].get_bfactor())

    return ca_atoms, residues, np.array(plddt)


# ----------------------------
# Align structures
# ----------------------------
def align_structures(exp_ca, af_ca, af_structure):
    n = min(len(exp_ca), len(af_ca))

    exp_ca = exp_ca[:n]
    af_ca = af_ca[:n]

    sup = Superimposer()
    sup.set_atoms(exp_ca, af_ca)

    # Apply transform to entire AF structure
    sup.apply(af_structure.get_atoms())

    return sup.rms, n


# ----------------------------
# Per-residue distances
# ----------------------------
def per_residue_distance(exp_ca, af_ca, n):
    d = []
    for i in range(n):
        d.append(np.linalg.norm(exp_ca[i].coord - af_ca[i].coord))
    return np.array(d)


# ----------------------------
# pLDDT plot
# ----------------------------
def plot_plddt(plddt):
    plt.figure()
    plt.plot(plddt, linewidth=2)
    plt.ylim(0, 100)
    plt.title("AlphaFold pLDDT (per residue confidence)")
    plt.xlabel("Residue index")
    plt.ylabel("pLDDT")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Distance plot
# ----------------------------
def plot_distances(dist):
    plt.figure()
    plt.plot(dist, marker="o", linewidth=1)
    plt.title("Per-residue CA deviation (Å)")
    plt.xlabel("Residue index")
    plt.ylabel("Distance (Å)")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Save PDB
# ----------------------------
def save_pdb(structure, filename):
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)


# ----------------------------
# PyMOL script generator
# ----------------------------
def write_pymol_script(exp_file, af_file, out="view.pml"):
    script = f"""
load {exp_file}, exp
load {af_file}, af

align af, exp

hide everything
show cartoon, exp
show cartoon, af

color blue, exp
color red, af

bg_color white
zoom
"""
    with open(out, "w") as f:
        f.write(script)

    print(f"PyMOL script written to {out}")


# ----------------------------
# NGLView visualization
# ----------------------------
def ngl_view(exp_file, af_file):
    v = nv.NGLWidget()

    v.add_component(exp_file)
    v.add_component(af_file)

    # Experimental = blue
    v[0].add_cartoon(color="blue")

    # AlphaFold = red
    v[1].add_cartoon(color="red")

    v.center()
    return v


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    print("Select experimental PDB")
    exp_path = select_file("Experimental PDB")

    print("Select AlphaFold PDB")
    af_path = select_file("AlphaFold PDB")

    exp = load_structure(exp_path)
    af = load_structure(af_path)

    exp_ca, exp_res, _ = extract_ca_and_plddt(exp)
    af_ca, af_res, plddt = extract_ca_and_plddt(af)

    rmsd, n = align_structures(exp_ca, af_ca, af)

    print(f"CA RMSD: {rmsd:.3f} Å")

    dist = per_residue_distance(exp_ca[:n], af_ca[:n], n)

    # Save aligned structures
    save_pdb(exp, "aligned_exp.pdb")
    save_pdb(af, "aligned_af.pdb")

    print("Saved aligned_exp.pdb and aligned_af.pdb")

    # Plots
    plot_distances(dist)
    plot_plddt(plddt[:n])

    # PyMOL script
    write_pymol_script("aligned_exp.pdb", "aligned_af.pdb")

    # NGLView
    view = ngl_view("aligned_exp.pdb", "aligned_af.pdb")

    print("Returning NGLView widget...")
    return view


if __name__ == "__main__":
    main()