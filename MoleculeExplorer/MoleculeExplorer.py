import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw


def analyze_smiles():
    smiles = smiles_entry.get().strip()

    if not smiles:
        messagebox.showwarning("Input Error", "Please enter a SMILES string.")
        return

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        messagebox.showerror("Invalid SMILES", "The SMILES string you entered is invalid.")
        return

    # Calculate properties
    exact_mw = Descriptors.ExactMolWt(mol)
    hbond_donors = Descriptors.NumHDonors(mol)
    tpsa = Descriptors.TPSA(mol)

    # Total formal charge: sum of all atom formal charges
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # Number of valence electrons
    valence_electrons = Descriptors.NumValenceElectrons(mol)

    # Update text info
    info_text = (
        f"SMILES: {smiles}\n"
        f"Exact Molecular Weight: {exact_mw:.4f}\n"
        f"Number of Hydrogen Bond Donors: {hbond_donors}\n"
        f"Topological Polar Surface Area (TPSA): {tpsa:.2f} Å²\n"
        f"Formal Charge: {formal_charge}\n"
        f"Number of Valence Electrons: {valence_electrons}"
    )
    info_label.config(text=info_text)

    # Generate molecule image
    img = Draw.MolToImage(mol, size=(300, 300))
    img_tk = ImageTk.PhotoImage(img)

    # Update image
    molecule_label.config(image=img_tk)
    molecule_label.image = img_tk  # keep reference


# Create main window
root = tk.Tk()
root.title("SMILES Molecule Analyzer")

# Input frame
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

tk.Label(input_frame, text="Enter SMILES:").pack(side=tk.LEFT)
smiles_entry = tk.Entry(input_frame, width=40)
smiles_entry.pack(side=tk.LEFT, padx=5)
tk.Button(input_frame, text="Analyze", command=analyze_smiles).pack(side=tk.LEFT)

# Molecule image
molecule_label = tk.Label(root)
molecule_label.pack(pady=10)

# Molecular info
info_label = tk.Label(root, text="", justify=tk.LEFT, font=("Arial", 12))
info_label.pack(pady=10)

root.mainloop()
